#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run pretraining sweep training...'
    
    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 1 \
        --eval_interval 1 \
        --finetune -1

    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 2 \
        --eval_interval 2 \
        --finetune -1
    
    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 10 \
        --eval_interval 10 \
        --finetune -1
     
    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 19 \
        --eval_interval 19 \
        --finetune -1

    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 48 \
        --eval_interval 24 \
        --finetune -1
    

    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 96 \
        --eval_interval 48 \
        --finetune -1

    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 1 \
        --eval_interval 1 \
        --finetune 1

    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 2 \
        --eval_interval 2 \
        --finetune 1
    
    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 10 \
        --eval_interval 10 \
        --finetune 1
     
    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 19 \
        --eval_interval 19 \
        --finetune 1

    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 48 \
        --eval_interval 24 \
        --finetune 1
    

    python train.py \
        --cuda \
        --data  /mnt/$2 \
        --checkpoints_dir /mnt/models \
        --dataset $2 \
        --model_size $4 \
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
        --local \
        --save_dir /mnt/wandb \
        --limit_train_batches 96 \
        --eval_interval 48 \
        --finetune 1
fi
