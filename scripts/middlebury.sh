#!/usr/bin/env bash
clear
set -x
DATAPATH="/data/slchen/data"

CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset middlebury \
    --datapath $DATAPATH --trainlist ./filenames/middlebury_train_list.txt \
    --testlist ./filenames/middlebury_train_h_list.txt \
    --batch_size  --test_batch_size 1 \
    --warmupepochs "1:1" \
    --epochs 200 --lr 0.0001 --lrepochs "100:10" \
    --save_freq 1 --sumr_freq 20 \
    --save_path "save/" \
    --model calnet-gc --logdir ./checkpoints/middle-t/calnet-gc \
    --load_ckpt checkpoints/em-1/calnet-gc/checkpoint_000282.ckpt 
    # > log/m-t.txt &