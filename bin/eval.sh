export SCORE_CONFIG='configs/score_config.yaml'
export MODEL_PATH='/home/data/v.moskvoretskii/cache/SCoRE/qwen_math_2st_16bs_klcorr_001_klinit_001_lr_5e-5_from_st1/checkpoint-50_full'
export DATA_PATH='/home/data/v.moskvoretskii/cache/SCoRE/qwen_math_2st_16bs_klcorr_001_klinit_001_lr_5e-5_from_st1/checkpoint-50_full/data'


python eval.py \
--config_path $SCORE_CONFIG \
--generation_model_path $MODEL_PATH \
--ft_dataset_path $DATA_PATH \
--iteration 0

