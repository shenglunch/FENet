
clear
set -x
# DATAPATH="/data/zh/data/sceneflow/"
DATAPATH="/data/yyx/data/sceneflow/"
# DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
# DATAPATH="/data1/zh/data/kitti/2012"
CUDA_VISIBLE_DEVICES=1 nohup python main.py --dataset sceneflow \
  --datapath $DATAPATH --trainlist ./filenames/sceneflow_final_train.txt \
  --testlist ./filenames/sceneflow_final_test.txt \
  --batch_size 6 --test_batch_size 1 \
  --warmupepochs "1:1" \
  --epochs 26 --lr 0.001 --lrepochs "12,16,20,24:2,2,2,2" \
  --save_freq 1 --sumr_freq 20 \
  --save_path "save/" \
  --model calnet-gc --logdir ./checkpoints/sceneflow-t/calnet-gc \
  --load_ckpt checkpoints/sceneflow-9/calnet-gc/checkpoint_000024.ckpt > log/sceneflow-24.txt &
  # > log/sceneflow-test1.txt &
