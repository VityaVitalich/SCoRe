import argparse
import os
import yaml
import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from functools import partial

from prompts import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.eval_utils import RewardEvaluator
from utils.generation_utils import load_config
from score_trainer_fsdp import SCoRETrainer
from copy import deepcopy
from trl import RLOOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
from dataclasses import dataclass, field


@dataclass
class SCoREConfig(RLOOConfig):
    stage: int = field(
        default=1,
        metadata={"help": "Stage of SCoRE Training"},
    )
    init_kl_coef: float = field(
        default=0.01,
        metadata={"help": "coef for initial generation KL"},
    )
    corr_kl_coef: float = field(
        default=0.01,
        metadata={"help": "coef for correction KL"},
    )


def custom_collate_fn(features, config, collator):
    # 1) Extract the text columns you want to keep
    #    but do NOT pass them to the default collator
    text_cols = [config['gold_col'], config['question_col']]
    batch_text = {}
    for col in text_cols:
        batch_text[col] = [f[col] for f in features]  # gather them as a list

    # 3) Use the default HF collator to pad the numeric parts only
    model_batch = collator([{'input_ids': f['input_ids']} for f in features])

    # 4) Attach the text lists back to the batch, as Python lists
    model_batch.update(batch_text)

    return model_batch

def add_input_ids(example, prompt_func):
    input_ids = prompt_func(example)
    example['input_ids'] = input_ids
    return example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to your config (YAML/JSON) with data_path, model_path, question_col, etc.")
    args = parser.parse_args()

    # 1) Load config
    config = load_config(args.config_path)



    score_dir = os.path.join(config['cache_dir'], 'SCoRE')
    os.makedirs(score_dir, exist_ok=True)
    run_dir = os.path.join(score_dir, config['run_name'])
    os.makedirs(run_dir, exist_ok=True)



    print(f"[INFO] Loading dataset from: {config['data_path']}")
    ds = datasets.load_from_disk(config['data_path'])

    # 3) Initialize the tokenizer
    print(f"[INFO] Loading tokenizer from: {config['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'],
        use_fast=True,
        cache_dir=config['cache_dir']

    )

    # 4) Load the prompt builder + few-shot
    prompt_builder = get_prompt_builder(config['task_type'])
    initial_generation_few_shot = load_few_shot_prompts(
        config['few_shot_dir'],
        f"{config['task_type']}_initial"
    )

    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        few_shot_prompts=initial_generation_few_shot,
        tokenize=True
    )

    ds = ds.map(partial(add_input_ids, prompt_func=initial_generation_prompt_func))

    ds['train'] = ds['train'].select_columns([config['question_col'], config['gold_col'], 'input_ids'])
    ds['test'] = ds['test'].select_columns([config['question_col'], config['gold_col'], 'input_ids'])

    reward_function = RewardEvaluator(config)

    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'],
        cache_dir=config['cache_dir'],
        torch_dtype=torch.bfloat16
    )
    ref_model = deepcopy(model)


    os.environ["WANDB_PROJECT"] = config['wandb_project_name']
    os.environ["WANDB_DIR"] = config['cache_dir']
    os.environ["WANDB_CACHE_DIR"] = config['cache_dir']
    
    score_config = SCoREConfig(
        output_dir=run_dir,
        exp_name=config['run_name'],
        seed=config['random_seed'],
        report_to='wandb',
        per_device_train_batch_size=config['per_device_train_batch_size'],
        local_rollout_forward_batch_size=config['local_rollout_forward_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        response_length=config['max_tokens'],
        temperature=config['temperature'],
        total_episodes=config['total_episodes'],
        num_sample_generations=0,
        corr_kl_coef=config['corr_kl_coef'],
        init_kl_coef=config['init_kl_coef'],
        save_steps=config['save_steps'],
        stage=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    trainer = SCoRETrainer(
        config=score_config,
        algo_config=config,
        processing_class=tokenizer,
        policy=model, 
        ref_policy=ref_model, 
        reward_model=reward_function, 
        train_dataset=ds["train"],
        data_collator=partial(custom_collate_fn, config=config, collator=DataCollatorWithPadding(tokenizer)),
        optimizers=(optimizer, None),
        prompt_builder=prompt_builder,
    )

    trainer.train()

if __name__ == "__main__":
    main()
