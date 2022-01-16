#!/usr/bin/env bash
clear
set -x
DATAPATH="/data/slchen/data"

# CUDA_VISIBLE_DEVICES=2,3 nohup python main.py --dataset em \
#     --datapath $DATAPATH --trainlist ./filenames/all_train.txt \
#     --testlist ./filenames/all_test.txt \
#     --batch_size 4 --test_batch_size 1 \
#     --warmupepochs "1:1" \
#     --epochs 400 --lr 0.001 --lrepochs "200,300:10,10" \
#     --save_freq 1 --sumr_freq 20 \
#     --save_path "save/" \
#     --model calnet-gc --logdir ./checkpoints/em-1/calnet-gc \
#     --load_ckpt checkpoints/sceneflow-9/calnet-gc/checkpoint_000024.ckpt > log/em-1.txt &
