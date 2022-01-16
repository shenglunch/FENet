#!/usr/bin/env bash
set -x
clear
DATAPATH="/data/slchen/data"
CUDA_VISIBLE_DEVICES=0,1 python save_mid.py --datapath $DATAPATH \
--testlist ./filenames/middlebury_train_h_list.txt \
--model calnet-gc \
--load_ckpt checkpoints/middle-6/calnet-gc/checkpoint_000149.ckpt