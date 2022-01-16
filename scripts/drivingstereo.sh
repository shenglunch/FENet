
clear
set -x
# DATAPATH="/data/zh/data/sceneflow/"
DATAPATH="/data/zh/data/DrivingStereo/"
# DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
# DATAPATH="/data1/zh/data/kitti/2012"
CUDA_VISIBLE_DEVICES=0,1 nohup python main.py --dataset drivingstereo \
  --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train_list.txt \
  --testlist ./filenames/drivingstereo_test_f_list.txt \
  --batch_size 1 --test_batch_size 1 \
  --warmupepochs "1:1" \
  --epochs 1 --lr 0.001 --lrepochs "2:10" \
  --save_freq 1 --sumr_freq 40 \
  --save_path "save/" \
  --model calnet-gc --logdir ./checkpoints/drivingstereo-t/calnet-gc \
  --load_ckpt checkpoints/drivingstereo-2/calnet-gc/checkpoint_000003.ckpt > log/drivingstereo-t.txt &
