# SCoRE

This repository contains code to reproduce the experiments from the paper **[Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/pdf/2409.12917)** using the **[TRL Framework](https://github.com/huggingface/trl/tree/main)** with DeepSpeed support. The reward functions and prompt builders are inspired by the **[Self-Taught Self-Correction for Small Language Models](https://arxiv.org/abs/2503.08681)** paper implementation.

A Docker container for running this repository is available at [vityavitalich/trl:score](https://hub.docker.com/repository/docker/vityavitalich/trl/tags/score/sha256-7e2ae6076271d08836c0c4ee112f6ca606d0bdb811c3d81e12beefe5c03cf75a).

## 🚀 Installation
### Option 1: Local Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd SCoRE
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Running Inside Docker (Recommended)
No need to install packages if using the Docker container. Just pull and run:
```bash
 docker pull vityavitalich/trl:score
 docker run -v $(pwd):/app -it vityavitalich/trl:score
```

## 🔧 Configuration Files
The configuration files are essential for defining the training setup, model parameters, and evaluation settings. The most important configuration file is `configs/score_config.yaml`, which defines all parameters needed for training.

## 💻 Running SCoRE
To start the training process, use the following commands:

### `bin/score.sh`
```bash
export ACCELERATE_CONFIG='configs/score_deepspeed.yaml'
export SCORE_CONFIG='configs/score_config.yaml'
export WANDB_API_KEY=''<YOUR_WANDB_API_KEY>

accelerate launch --config_file $ACCELERATE_CONFIG score.py --config_path $SCORE_CONFIG
```

## 📊 Parameters Description

### Model Configuration
| Parameter       | Description                          | Default Value                     |
--------------|-------------------------------------|----------------------------------|
| `model_path`    | Path to pretrained model checkpoint.| `Qwen/Qwen2.5-Math-1.5B-Instruct`|
| `cache_dir`     | Directory to cache model weights.   | `/home/data/v.moskvoretskii/cache/`|
| `random_seed`   | Seed for reproducibility.           | `42`                              |
| `wandb_project_name` | Project name for logging.      | `SCoRE`                           |

### Dataset Configuration
| Parameter       | Description                          | Default Value |
|-----------------|-------------------------------------|---------------|
| `task_type`     | Task type, e.g., `math` or `qa`.    | `math`        |
| `data_path`     | Path to the dataset directory.      | `data/math500`|
| `id_col`        | Unique identifier column.           | `unique_id`   |
| `question_col`  | Column containing questions.        | `problem`     |
| `gold_col`      | Column with reference answers.      | `answer`      |

### Reward Function Configuration

The reward function is responsible for evaluating generated answers based on specific criteria. For `math` tasks, the `evaluator_mode` is always set to `final` and cannot be changed. For `qa` tasks, both `default` and `final` modes are supported.

| Parameter              | Description                                                                                         | Default Value                       |
|------------------------|---------------------------------------------------------------------------------------------------|------------------------------------|
| `evaluator_mode`       | Specifies how the generated answer is evaluated. `default` evaluates the entire generation, while `final` evaluates only the portion after a specific keyword defined by `evaluator_answer_marker`. **Note:** For `math` tasks, this is always set to `final`. | `final` |
| `evaluator_function`   | The metric used for evaluation. Options include: `math_acc` (for math tasks), `in_acc`, `f1`, `em` (for QA tasks). | `math_acc` |
| `evaluator_answer_marker` | A keyword or phrase indicating where the final answer starts in the generated text. Text preceding this marker is ignored during evaluation. Example: `Final Answer: The final answer is` for mathematical tasks. | `Final Answer: The final answer is` |


### Generation Configuration
| Parameter                     | Description                         | Default Value |
|------------------------------|------------------------------------|---------------|
| `few_shot_dir`               | Directory for few-shot learning examples. | `few_shots`    |
| `number_output_initial_generations` | Number of answers generated per prompt. | `1`           |
| `temperature`                | Sampling temperature for randomness.     | `0.9`         |
| `max_tokens`                 | Maximum tokens generated per prompt.    | `1024`        |

### Training Configuration
| Parameter                          | Description                                   | Default Value |
|-----------------------------------|----------------------------------------------|---------------|
| `per_device_train_batch_size`      | Number of samples per batch for each GPU.    | `1`           |
| `gradient_accumulation_steps`      | Steps to accumulate gradients.              | `4`           |
| `local_rollout_forward_batch_size` | Batch size for multiple rollouts.           | `4`           |
| `total_episodes`                   | Total training samples processed.           | `100`         |
| `learning_rate`                    | Learning rate for training.                 | `5.0e-5`      |
| `num_warmup_steps`                 | Warmup steps for learning rate scheduler.   | `100`         |
| `save_steps`                       | Checkpoint saving interval.                 | `1`           |

### LoRA Configuration
| Parameter     | Description                         | Default Value |
|---------------|-----------------------------------|---------------|
| `use_lora`    | Whether to use LoRA.              | `True`        |
| `lora_rank`   | Rank of LoRA adaptation.          | `32`          |
| `lora_alpha`  | Scaling factor for LoRA.          | `8`           |
| `lora_dropout`| Dropout rate for LoRA layers.     | `0.1`         |

## 📖 References
- [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/pdf/2409.12917)
- [Self-Taught Self-Correction for Small Language Models](https://arxiv.org/abs/2503.08681)


---

## 📚 Citation

If you use this repository, please cite it as follows:

```bibtex
@misc{SCoRE2025,
  author = {Viktor Moskvoretskii},
  title = {SCoRE: Open-Source Implementation of Training Language Models to Self-Correct via Reinforcement Learning},
  year = {2025},
  url = {https://github.com/VityaVitalich/SCoRe},
  note = {Accessed: YYYY-MM-DD}
}
```
