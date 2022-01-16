#!/usr/bin/env bash
clear
set -x
DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
# DATAPATH="/data/zh/data/kitti/2015/data_scene_flow/"

CUDA_VISIBLE_DEVICES=2,3 nohup python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt \
    --batch_size 4 --test_batch_size 4 \
    --lr 0.00001 \
    --epochs 400 \
    --warmupepochs "10, 20, 30, 40:3.0, 3.0, 3.0, 3.0" \
    --lrepochs "100, 125, 150, 175, 200, 225, 250, 275, 300:1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5" \
    --model gwcnet-gc \
    --logdir ./checkpoints/kitti15-3/gwcnet-gc \
    --loadckpt ./checkpoints/sceneflow-rcml-2-1/rsml-2-1-checkpoint_000020.ckpt > log/kitti15-3.txt &


# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset kitti \
#     --datapath $DATAPATH --trainlist ./filenames/xxkitti15_train.txt --testlist ./filenames/kitti15_val.txt \
#     --batch_size 8 --test_batch_size 8 \
#     --epochs 193 --lr 0.001 --lrepochs "200:10" \
#     --model gwcnet-gc \
#     --save_path "save/" \
#     --logdir ./checkpoints/kitti15-test/gwcnet-gc \
#     --loadckpt ./checkpoints/sceneflow-rc-1/gwcnet-gc/checkpoint_000024.ckpt
    # checkpoints/sceneflow-1-4-15/gwcnet-gc/checkpoint_000021.ckpt
    