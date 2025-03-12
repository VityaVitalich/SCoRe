export ACCELERATE_CONFIG='configs/score_deepspeed.yaml' #'configs/ddp_config.yaml'  #'configs/accelerate_config.yaml'
export SCORE_CONFIG='configs/score_config.yaml'
export WANDB_API_KEY=''


accelerate launch --config_file $ACCELERATE_CONFIG score.py --config_path $SCORE_CONFIG
