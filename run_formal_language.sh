#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    echo $2
    python train.py \
        --cuda \
        --data /datadrive/formal_language/$2/ \
        --dataset $2 \
        --model_size $3 \
        --n_positions 500 \
        --n_ctx 128 \
        --n_layer 2 \
        --d_model 64 \
        --n_head 2 \
        --d_ff 4 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.0025 \
        --warmup_step 0 \
        --max_step 400000 \
        --batch_size 4 \
        --gpu0_bsz 1 \
        --max_epoch 1
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
