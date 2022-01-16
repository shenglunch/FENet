#!/usr/bin/env bash
clear
set -x
DATAPATH="/data/yyx/data/kitti"

CUDA_VISIBLE_DEVICES=1 nohup python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti_all_train.txt \
    --testlist ./filenames/kitti_all_val.txt \
    --batch_size 6 --test_batch_size 1 \
    --warmupepochs "1:1" \
    --epochs 1 --lr 0.001 --lrepochs "500,750:10,10" \
    --save_freq 1 --sumr_freq 20 \
    --save_path "save/" \
    --model calnet-gc --logdir ./checkpoints/kitti-t/calnet-gc \
    --load_ckpt checkpoints/kitti-all-3/calnet-gc/checkpoint_000996.ckpt > log/kitti-t.txt &
  
