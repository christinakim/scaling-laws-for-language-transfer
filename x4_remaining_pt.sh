#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run pretraining sweep training...'
    
    python train.py \
        --cuda \
        --data  /home/christina/$2 \
        --checkpoints_dir /home/christina/models \
        --dataset $2 \
        --model_size x4small \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --warmup_step 300 \
        --max_step 250000 \
        --n_gpus $3  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --eval_interval 240 \
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 480 \
        --finetune -1
    
    python train.py \
        --cuda \
        --data  /home/christina/$2 \
        --checkpoints_dir /home/christina/models \
        --dataset $2 \
        --model_size x4small \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --warmup_step 300 \
        --max_step 250000 \
        --n_gpus $3  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --eval_interval 480 \
        --limit_train_batches 960 \
        --local \
        --save_dir /mnt/wandb \
        --finetune 1
fi