#!/usr/bin/env bash
clear
set -x
DATAPATH="/data/yyx/data/kitti/2012"

# CUDA_VISIBLE_DEVICES=1,2,3 nohup python main.py --dataset kitti \
#     --datapath $DATAPATH --trainlist ./filenames/kitti12_train.txt \
#     --testlist ./filenames/kitti12_val.txt \
#     --batch_size 6 --test_batch_size 1 \
#     --warmupepochs "1:1" \
#     --epochs 800 --lr 0.001 --lrepochs "700:10" \
#     --save_freq 1 --sumr_freq 10 \
#     --save_path "save/" \
#     --model calnet-gc --logdir ./checkpoints/kitti12-aug/calnet-gc \
#     --load_ckpt checkpoints/drivingstereo/calnet-gc/checkpoint_000002.ckpt > log/kitti12-aug.txt &
  
