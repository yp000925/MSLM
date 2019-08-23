#!/usr/bin/env bash

#IMAGE_ROOT=~/fyp/vehicle-triplet-reid/VeRi_with_plate/image_train ; shift
IMAGE_ROOT=../vehicle-triplet-reid/VeRi_with_plate/image_train ; shift
#INIT_CHECKPT=./pre_trained_model/resnet_v1_50.ckpt; shift
INIT_CHECKPT=../vehicle-triplet-reid/pre_trained_model/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt; shift

EXP_ROOT=./experiments/VeRi776_merge ; shift


python train.py \
    --train_set ../vehicle-triplet-reid/VeRi_with_plate/veri_train.txt \
    --model_name mobilenet_v1_1_224 \
    --head_name fusion\
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --detailed_logs \
    --embedding_dim 128 \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 224 --pre_crop_width 224 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric euclidean \
    --loss merged_loss \
    --learning_rate 3e-4 \
    --train_iterations 60000 \
    --decay_start_iteration 30000 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
#    --detailed_logs \
    "$@"
