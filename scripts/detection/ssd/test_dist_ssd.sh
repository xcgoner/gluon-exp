#!/bin/bash 
# source activate python2
# cd /home/ubuntu/mxnet-dist-exp/example/ssd
# python train.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 32 --kv-store dist_sync_allreduce --network resnet50
export CUDA_VISIBLE_DEVICES=5,6,7,8
python /home/ubuntu/gluon-exp/scripts/detection/ssd/train_ssd.py --gpus $OMPI_COMM_WORLD_RANK --kv-store dist_sync_allreduce