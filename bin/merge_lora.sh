#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
ADAPTERS_PATH="/home/data/v.moskvoretskii/cache/SCoRE/qwen_math_2st_16bs_klcorr_001_klinit_001_lr_5e-5/"
CACHE_DIR="/home/data/v.moskvoretskii/cache/"

python merge_lora.py \
--base_model=$BASE_MODEL \
--adapters_path=$ADAPTERS_PATH \
--cache_dir=$CACHE_DIR
