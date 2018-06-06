#!/bin/bash 
# source activate python2
# cd /home/ubuntu/mxnet-dist-exp/example/ssd
# python train.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 32 --kv-store dist_sync_allreduce --network resnet50
export CUDA_VISIBLE_DEVICES=4,5,6,7
# python /home/ubuntu/gluon-exp/scripts/detection/ssd/train_ssd.py --gpus $OMPI_COMM_WORLD_RANK --kv-store dist_sync
# python /home/ubuntu/gluon-exp/scripts/detection/ssd/train_ssd.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 32 --network vgg16_atrous --data-shape 300 --dataset voc --lr 0.001 --lr-decay-epoch 160,200 --lr-decay 0.1 --epochs 240 --kv-store dist_sync
# python /home/ubuntu/gluon-exp/scripts/detection/ssd/train_ssd.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 32 --network vgg16_atrous --data-shape 300 --dataset voc --lr 0.001 --lr-decay-epoch 160,200 --lr-decay 0.1 --epochs 240 --kv-store dist_sync
# python /home/ubuntu/gluon-exp/scripts/detection/ssd/train_ssd.py --gpus 0,1,2,3 --kv-store device
# python /home/ubuntu/incubator-mxnet/example/gluon/image_classification.py --model vgg11 --gpus $OMPI_COMM_WORLD_RANK --kvstore dist_sync
# python /home/ubuntu/incubator-mxnet/example/image-classification/train_cifar10.py --network resnet --num-layers 110 --batch-size 128 --gpus $OMPI_COMM_WORLD_RANK --kv-store dist_sync
python /home/ubuntu/gluon-exp/scripts/detection/ssd/train_dist_ssd.py --gpus $OMPI_COMM_WORLD_RANK --batch-size 64 --network resnet50_v1 --data-shape 512 --dataset voc --lr 0.001 --lr-decay-epoch 160,200 --lr-decay 0.1 --epochs 240 --kv-store dist_sync

# stdbuf -o 0 python script 2>&1 | tee filename
