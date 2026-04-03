#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./train.py \
        --batch_size 64 \
        --accumulate_gradients 2 \
        --amp 0 \
        --dump_path ./dump \
        --max_input_dimension 10 \
        --exp_name vae \
        --exp_id run-train \
        --lr 1e-3 \
        --latent_dim 512 \
        --save_periodic 10 \
        --n_steps_per_epoch 2000 \
        --max_epoch 500 \
        --kl_limits 1.0 \
        --model_type vae \
        --wandb_run_name "train-example" \
        --wandb_group_name "training" \
        --wandb_project "symbolic-regression-training" \
        --print_freq 100
