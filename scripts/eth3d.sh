#!/usr/bin/env bash
clear
set -x
DATAPATH="/data/slchen/data"

CUDA_VISIBLE_DEVICES=1 nohup python main.py --dataset eth3d \
    --datapath $DATAPATH --trainlist ./filenames/eth3d_train_list.txt \
    --testlist ./filenames/eth3d_val_list.txt \
    --batch_size 4 --test_batch_size 1 \
    --warmupepochs "1:1" \
    --epochs 200 --lr 0.0001 --lrepochs "100:10" \
    --save_freq 1 --sumr_freq 20 \
    --save_path "save/" \
    --model calnet-gc --logdir ./checkpoints/eth3d-3/calnet-gc \
    --load_ckpt checkpoints/eth3d-3/calnet-gc/checkpoint_000197.ckpt > log/e-3.txt &