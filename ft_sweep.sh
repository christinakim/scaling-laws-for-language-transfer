#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run finetuning sweep training...'
    python train.py \
        --cuda \
        --data  /home/christina/$2 \
        --checkpoints_dir /home/christina/models \
        --dataset $2 \
        --model_size x6small \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $3  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --eval_interval 1 \
        --limit_train_batches 1 \
        --finetune 1
    
    python train.py \
        --cuda \
        --data  /home/christina/$2 \
        --checkpoints_dir /home/christina/models \
        --dataset $2 \
        --model_size x6small \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $3  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --eval_interval 2 \
        --limit_train_batches 2 \
        --finetune 1
    
    python train.py \
        --cuda \
        --data  /home/christina/$2 \
        --checkpoints_dir /home/christina/models \
        --dataset $2 \
        --model_size x6small \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $3  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --eval_interval 20 \
        --limit_train_batches 20 \
        --finetune 1
    
    python train.py \
        --cuda \
        --data  /home/christina/$2 \
        --checkpoints_dir /home/christina/models \
        --dataset $2 \
        --model_size x6small \
        --n_ctx 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --clip 1.0 \
        --batch_size 512 \
        --mini_batch_size 8 \
        --warmup_step 500 \
        --max_step 250000 \
        --n_gpus $3  \
        --eval_batch_size 2 \
        --max_eval_steps 2 \
        --max_epoch 10000 \
        --eval_interval 24 \
        --limit_train_batches 48 \
        --finetune 1
    

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
else
    echo 'unknown argment 1'
fi
