#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_2.py \
        --cuda \
        --data /datadrive/$2 \
        --dataset $2 \
        --model_size $3 \
        --n_positions 500 \
        --n_ctx 256 \
        --n_layer 2 \
        --d_model 64 \
        --n_head 2 \
        --d_ff 4 \
        --lr 0.0025 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --warmup_step 0 \
        --max_step 50000 \
        --batch_size 32 \
        --gpu0_bsz 1 \
        --eval_interval 100\
        --n_nodes 1 \
        --n_gpus 1  \
        --eval_batch_size 5 \
        --max_eval_steps 100 \
        --max_epoch 100

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
else
    echo 'unknown argment 1'
fi
