#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data /datadrive/$2/ \
        --dataset $2 \
        --model_size x8small \
        --n_positions 500 \
        --n_ctx 128 \
        --optim adam \
        --warmup_step 0 \
        --max_step 400000 \
        --batch_size 4 \
        --eval_interval 4000\
        --max_epoch 1
    python train.py \
        --cuda \
        --data /datadrive/$2/ \
        --dataset $2 \
        --model_size x6small \
        --n_positions 500 \
        --n_ctx 128 \
        --optim adam \
        --warmup_step 0 \
        --max_step 400000 \
        --batch_size 4 \
        --eval_interval 4000\
        --max_epoch 1
     python train.py \
        --cuda \
        --data /datadrive/$2/ \
        --dataset $2 \
        --model_size x4small \
        --n_positions 500 \
        --n_ctx 128 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 400000 \
        --batch_size 4 \
        --eval_interval 4000\
        --max_epoch 1
    python train.py \
        --cuda \
        --data /datadrive/$2/ \
        --dataset $2 \
        --model_size x2small \
        --n_positions 500 \
        --n_ctx 128 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 400000 \
        --batch_size 4 \
        --eval_interval 4000\
        --max_epoch 1
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
else
    echo 'unknown argment 1'
fi
