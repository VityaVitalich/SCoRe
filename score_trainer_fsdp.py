import gc
import math
import os
import time
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    get_cosine_schedule_with_warmup
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

# Assume these come from TRL’s code (as in RLOOTrainer):
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)
from trl.trainer.rloo_config import RLOOConfig  # or create a new SCoREConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, log_table_to_comet_experiment
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from copy import deepcopy


# TODO: Add Scheduler, add validation

INVALID_LOGPROB = 1.0

class SCoRETrainer(Trainer):
    """
    A single-stage SCoRE algorithm implemented in the style of TRL's RLOOTrainer.

    Key differences from standard PPO/POLICY GRAD approaches:
      - We generate an INITIAL answer (which gets a KL penalty vs. ref policy).
      - Then we generate a CORRECTION from the policy, which gets a reward from
        the reward/cost function.
      - Final scalar = Reward(correction) - beta * KL(initial).
      - Use REINFORCE on the correction tokens to update the policy.

    Everything else (Accelerator, logging, etc.) is kept consistent
    with the RLOOTrainer style.
    """
    _tag_names = ["trl", "score"]  # for optional tracking in your model config

    def __init__(
        self,
        config: RLOOConfig,
        algo_config,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        prompt_builder,
        reward_model: Union[nn.Module, Callable[[list[str]], list[float]]],
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
        """
        Very similar to RLOOTrainer.__init__, except we note that we only do single-step REINFORCE
        logic inside the train loop.
        """
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, pass a *copy* or pass `None` if using PEFT’s read-only approach."
            )

        self.args = config  # For TRL, a config derived from RLOOConfig or similar
        args = config
        self.algo_config = algo_config
        self.processing_class = processing_class
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # used by HF Trainer if re-creating optimizers
        self.prompt_builder = prompt_builder

        # If no collator provided, create a default one
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)
        self.data_collator = data_collator

        # Remove dropout from policy/ref/reward if desired
        for module in [policy, ref_policy, reward_model]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)

        # Setup generation config for initial answer vs. correction
        # (You can store them in self.args or define them below)
        self.init_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,  # or some separate param, e.g. args.initial_answer_length
            temperature=args.temperature,
            #top_k=0,
            top_p=0.8,
            do_sample=True,
            pad_token_id=None,
            eos_token_id=None,
        )
        self.corr_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,  # or separate param, e.g. correction_length
            temperature=args.temperature,
            #top_k=0,
            top_p=0.8,
            do_sample=True,
            pad_token_id=None,
            eos_token_id=None,
        )

        # Construct the dataloader
        self.train_dataset_len = len(train_dataset)
        if args.total_episodes is None:  # allow user to define episodes in terms of epochs
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)

        # Build accelerator
        accelerator = Accelerator()
        self.accelerator = accelerator
        self.accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = self.algo_config['gradient_accumulation_steps']
        args.world_size = accelerator.num_processes

        # This part is from RLOO: computing local_batch_size, micro_batch_size, etc.
        args.local_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        # we do not do multiple mini-batches in this example, so skip that part

        # total number of train steps
        args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)
        # name runs etc.
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()
        args.run_name = f"{args.exp_name}"

        # Seeds, directories, etc.
        self.local_seed = args.seed
        torch.manual_seed(args.seed)

        # Prepare data loader
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        self.model = policy  # HF Trainer expects self.model
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.algo_config['num_warmup_steps'], num_training_steps=args.num_total_batches)


        # reset local seed
        torch.manual_seed(self.local_seed)

        # Prepare eval dataloader if needed
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=True,
            )
            self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
        else:
            self.eval_dataloader = None

        # If using DeepSpeed / FSDP
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        if self.is_deepspeed_enabled:
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = self.reward_model.to(self.accelerator.device)

        # Create optimizer if not passed in
        if self.optimizer is None:
            self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        # Setup HF Trainer state + callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)


        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None

        # Create local dir, push to hub, etc.
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Tag model if needed
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def train(self):
        """
        Single-stage SCoRE training loop:
          1) Generate initial answer (and compute KL vs. ref policy).
          2) Generate correction (and get reward from reward_model).
          3) Final reward = reward(correction) - beta * KL(initial).
          4) REINFORCE update on the correction’s log-probs.
        """
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device
        dataloader = self.dataloader

        # internal trainer states
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches  # or something else
        self.state.num_train_epochs = args.num_train_epochs  # for logging

        # Start
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        start_time = time.time()

        # Reusable function to get next batch, infinitely
        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        self.model.train()

        for step_idx in range(1, args.num_total_batches + 1):
            data = next(iter_dataloader)
            self.state.episode += args.batch_size

            # ------------------------
            # 1) Generate INITIAL
            # ------------------------
            with torch.no_grad():
                queries = data["input_ids"].to(device).long()

                # Possibly also pass "attention_mask" or other keys
                # Generate the initial answer
                with unwrap_model_for_generation(
                    self.model, accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation
                ) as unwrapped_model:
                    
                    init_outputs, init_logits = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        self.processing_class.pad_token_id,
                        self.init_generation_config,
                    )

                    # We store only the portion beyond the prompt length
                    context_len = queries.shape[1]
                    init_answers = init_outputs[:, context_len:]

                    # logp(policy) for init
                    init_logprob = selective_log_softmax(init_logits, init_answers)
                    # logp(ref) for init
                    ref_outputs = forward(self.ref_policy, init_outputs, self.processing_class.pad_token_id)
                    ref_logits = ref_outputs.logits[:, context_len - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, init_answers)

                    # Sum across tokens for the KL
                    mask = (init_answers == self.processing_class.pad_token_id)
                    init_logprob = init_logprob.masked_fill_(mask, 0)
                    ref_logprob = ref_logprob.masked_fill_(mask, 0)
                    kl_init = (init_logprob - ref_logprob).sum(dim=1)

                    del ref_logprob, ref_logits, init_logprob, init_logits, init_outputs, queries
                    torch.cuda.empty_cache()

                # ------------------------
                # 2) Generate CORRECTION
                # ------------------------
                    init_answer_texts = self.processing_class.batch_decode(init_answers, skip_special_tokens=False)
                    # print('===init answers===')
                    # print(device, init_answer_texts)
                    corr_inputs = build_correction_inputs_for_batch(
                        data, 
                        init_answer_texts,
                        self.processing_class,
                        self.prompt_builder,
                        question_col=self.algo_config['question_col'],
                    ).to(device)
                
                #print('===correction inputs===')
                #print(device, self.processing_class.batch_decode(corr_inputs.cpu(), skip_special_tokens=False))
                
                       
                    corr_outputs, corr_logits = batch_generation(
                        unwrapped_model,
                        corr_inputs,
                        args.local_rollout_forward_batch_size,
                        self.processing_class.pad_token_id,
                        self.init_generation_config,
                    )
            
            # print('===correction inputs===')
            # print(device, self.processing_class.batch_decode(corr_outputs, skip_special_tokens=False))
            
            # The correction is the portion after the entire corr_inputs length
            corr_context_len = corr_inputs.shape[1]
            corr_tokens = corr_outputs[:, corr_context_len:]

            del corr_logits
            torch.cuda.empty_cache()
            gc.collect()

            with torch.no_grad():
                # If your reward model is a python function that takes strings
                corr_output_text = self.processing_class.batch_decode(corr_tokens, skip_special_tokens=True)
                
                reward_vals = [self.reward_model(
                        model_answer=corr_output,
                        ground_truth=reference
                        ) 
                        for (corr_output, reference) in zip(corr_output_text, data[self.algo_config['gold_col']]
                        )
                        ]
                reward_vals = torch.tensor(
                    reward_vals,
                    dtype=torch.float,
                    device=device
                )
                reward_vals = reward_vals.reshape(-1)

                #print(reward_vals)

            # final scalar reward
            final_reward = reward_vals - args.kl_coef * kl_init  # rename "kl_coef" or "beta" as needed

            #print(kl_init.float())

            # ------------------------
            # 3) REINFORCE on correction
            # ------------------------
            # We'll do a forward pass for the correction tokens to get log p_theta(correction).
            # Then do loss = - final_reward * sum_{tokens}( log p_theta ).
            # We can do it in micro-batches if needed. For simplicity, do it in one pass:
            # But we still use `accelerator.accumulate(model)` if gradient_accum_steps>1.


            # or if you want to chunk further
            for micro_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
                with accelerator.accumulate(self.model):
                    micro_end = micro_start + args.per_device_train_batch_size
                    mb_idx = slice(micro_start, micro_end)
                    mb_corr_outputs = corr_outputs[mb_idx]
                    mb_corr_tokens = corr_tokens[mb_idx]
                    mb_final_reward = final_reward[mb_idx]

                    # forward pass
                    out = forward(self.model, mb_corr_outputs, self.processing_class.pad_token_id)
                    # slice the region that corresponds to correction tokens
                    # If the correction started at corr_outputs.shape[1] - corr_len:
                    # we align with logits. Typically we do "logits[:, :-1]" vs. "tokens[:, 1:]",
                    # but we keep it consistent with your code:
                    offset = mb_corr_outputs.shape[1] - mb_corr_tokens.shape[1] - 1
                    logits_corr = out.logits[:, offset:-1, :]
                    logits_corr /= args.temperature + 1e-7
                    logprob_corr = selective_log_softmax(logits_corr, mb_corr_tokens)

                    # sum across time
                    sum_lp = logprob_corr.sum(dim=1)
                    # negative sign => we want to maximize => so we minimize negative
                    loss = -(mb_final_reward * sum_lp).mean()

                    # Backprop
                    accelerator.backward(loss)
                    self.optimizer.step()


                self.lr_scheduler.step()



            # ------------------------
            # Logging / stats
            # ------------------------
            with torch.no_grad():
                # e.g. gather stats for logging
                mean_kl_init = accelerator.gather_for_metrics(kl_init).mean().item()
                mean_reward = accelerator.gather_for_metrics(reward_vals).mean().item()
                metrics = {}
                metrics["score/kl_init"] = mean_kl_init
                metrics["score/reward"] = mean_reward
                metrics["score/final_reward"] = accelerator.gather_for_metrics(final_reward).mean().item()
                metrics["loss"] = loss.item()
                metrics["episode"] = self.state.episode
                metrics["step"] = step_idx
                # log
                self.log(metrics)

            del corr_outputs, corr_tokens, out, offset, logits_corr, logprob_corr, sum_lp, final_reward, kl_init
            torch.cuda.empty_cache()
            gc.collect()

            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            # if self.control.should_save:
            #     #with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(self.model):
            #     self._save_checkpoint(self.model, trial=None)
            #     self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            # Optionally sample completions or do evaluations
            if (
                args.num_sample_generations > 0
                and (step_idx - 1) % max(1, args.num_total_batches // args.num_sample_generations) == 0
            ):
                self.generate_completions(sampling=True)

            # Early termination if needed
            if self.control.should_training_stop:
                break

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            #with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(self.model):
            self._save_checkpoint(self.model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        print("SCoRE training completed!")

    def generate_completions(self, sampling: bool = False):
        """
        Utility function to sample model completions on `eval_dataloader` and log them.
        Copied from RLOOTrainer's style, but simplified.
        """

        raise NotImplementedError('Not Yet')

        # args = self.args
        # if self.eval_dataloader is None:
        #     return

        # generation_config = GenerationConfig(
        #     max_new_tokens=args.response_length,
        #     temperature=(0.01 + 1e-7),
        #     top_k=0.0,
        #     top_p=1.0,
        #     do_sample=True,
        # )

        # table = defaultdict(list)
        # with unwrap_model_for_generation(
        #     self.model, self.accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation
        # ) as unwrapped_model:
        #     for batch in self.eval_dataloader:
        #         query = batch["input_ids"]
        #         with torch.no_grad():
        #             context_length = query.shape[1]
        #             query_response, _ = batch_generation(
        #                 unwrapped_model,
        #                 query,
        #                 query.shape[0],
        #                 self.processing_class.pad_token_id,
        #                 generation_config,
        #             )
        #             response = query_response[:, context_length:]
        #             table["query"].extend(
        #                 gather_object(self.processing_class.batch_decode(query, skip_special_tokens=True))
        #             )
        #             table["model response"].extend(
        #                 gather_object(self.processing_class.batch_decode(response, skip_special_tokens=True))
        #             )

        #         if sampling:
        #             # Just do one batch if sampling
        #             break

        # df = pd.DataFrame(table)
        # if self.accelerator.is_main_process:
        #     print_rich_table(df.iloc[0 : 0 + 5])
        #     # If using W&B or Comet, you can log the table
        #     # ...
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SCoRE",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))



def build_correction_inputs_for_batch(
    batch,
    init_answer_texts,
    tokenizer,
    prompt_builder,
    question_col: str = "question",
    initial_answer_col: str = "initial_answer",
):
    # We will store the final "correction input" for each row
    batch_correction_inputs = []

    for i, init_ans_text in enumerate(init_answer_texts):
        question_text = batch[question_col][i]

        # Build a 'sample' dict as your prompt_builder expects:
        sample_for_prompt = {
            question_col: question_text,
            initial_answer_col: [init_ans_text],  # your builder uses a list for initial answers
        }

        # Now get the final correction prompt(s). Typically there's 1 prompt
        # in this scenario, but build_correction_prompt returns a list.
        corr_inputs = prompt_builder.build_correction_prompt(
            sample=sample_for_prompt,
            tokenizer=tokenizer,
            question_col=question_col,
            initial_answer_col=initial_answer_col,
            tokenize=True
        )
        # Since we used only 1 initial answer, corr_prompts[0] is the final text
        corr_inputs = corr_inputs[0]

        batch_correction_inputs.append({'input_ids': corr_inputs})


    collated_corrections = DataCollatorWithPadding(tokenizer)(batch_correction_inputs)
    return collated_corrections['input_ids']

