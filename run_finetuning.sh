#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data /datadrive/$2 \
        --dataset $2 \
        --model_size $3 \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --eval_interval 10 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $4  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 100 \
        --local \
        --finetune 1 \
        --note $5 \
        --limit_train_batches $6


elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
else
    echo 'unknown argment 1'
fi
