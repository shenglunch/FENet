#!/usr/bin/env bash
set -x
clear
DATAPATH="/data/yyx/data/kitti/2012"
CUDA_VISIBLE_DEVICES=2 python save_disp.py --datapath $DATAPATH \
--testlist ./filenames/kitti12_test.txt \
--model calnet-gc \
--load_ckpt checkpoints/kitti-all-3/calnet-gc/checkpoint_000996.ckpt