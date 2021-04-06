#!/bin/bash
MODEL_SIZES="x6small x5small x4small x3small x2small small"

for model in $MODEL_SIZES
do
    python train.py \
        --cuda \
        --data  /$1/$2 \
        --checkpoints_dir /$1/models \
        --dataset $2 \
        --model_size $3 \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 1 \
        --mini_batch_size 8 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $4  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --local \
        --save_dir /$1/wandb \
        --eval_interval $5 \
        --limit_train_batches $5 \
        --finetune -1

    python train.py \
        --cuda \
        --data  /$1/$2 \
        --checkpoints_dir /$1/models \
        --dataset $2 \
        --model_size $3 \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 1 \
        --mini_batch_size 8 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $4  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --local \
        --save_dir /$1/wandb \
        --eval_interval $5 \
        --limit_train_batches $5 \
        --finetune 1
done