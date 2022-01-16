#!/usr/bin/env bash
set -x
clear
DATAPATH="/data/slchen/data"
CUDA_VISIBLE_DEVICES=1 python save_eth3d.py --datapath $DATAPATH \
--testlist ./filenames/eth3d_test_list.txt \
--model calnet-gc \
--load_ckpt checkpoints/eth3d-3/calnet-gc/checkpoint_000197.ckpt