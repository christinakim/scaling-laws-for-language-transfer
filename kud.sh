#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data  /home/christina/$2 \
        --checkpoints_dir /home/christina/models \
        --dataset $2 \
        --model_size $3 \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $4  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --note $5 \
        --eval_interval $6 \
        --limit_train_batches $7 \
        --finetune $8


elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
else
    echo 'unknown argment 1'
fi
