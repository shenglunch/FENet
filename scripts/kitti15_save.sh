#!/usr/bin/env bash
clear
set -x
DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
CUDA_VISIBLE_DEVICES=1 python save_disp.py --datapath $DATAPATH \
  --testlist filenames/kitti15_test.txt \
  --model calnet-gc \
  --load_ckpt checkpoints/kitti-all-3/calnet-gc/checkpoint_000996.ckpt
