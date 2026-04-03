#!/bin/bash

MODEL_DIR="./weights"
MODEL_CHECKPOINT="checkpoint.pth"

DATA_TYPE="feynman"

CUDA_VISIBLE_DEVICES=0 python -u ./LSO_eval.py --reload_model_dir "${MODEL_DIR}" --reload_model  "${MODEL_CHECKPOINT}" --eval_lso_on_pmlb True --pmlb_data_type "${DATA_TYPE}" --target_noise 0.0 --max_input_points 200 --lso_optimizer es_fromvae_fit --beam_size 2 --model_type vae --lso_stop_r2 0.95 --compre_num 128 --lso_stop_r2_abandon_lower 0 --feynman_sel_equs_num -1 --com_weight 400 --ev_sigma 1.3 --warmup_iteration 15 --pop_init_index 0*1*1 --num_eval_workers 16 --wandb_project symbolic-regression-example --wandb_group_name "eval-${DATA_TYPE}" --wandb_run_name pop80_iter120 --pop_num 80 --mu_num 16 --lso_max_iteration 120