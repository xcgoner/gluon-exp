#!/bin/bash 
# source activate python2
# cd /home/ubuntu/mxnet-dist-exp/example/ssd
# python train.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 32 --kv-store dist_sync_allreduce --network resnet50
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /home/ubuntu/gluon-exp/scripts/detection/ssd/train_ssd.py --gpus $OMPI_COMM_WORLD_RANK --kv-store dist_sync_allreduce

# stdbuf -o 0 python script 2>&1 | tee filename