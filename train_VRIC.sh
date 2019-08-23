#!/usr/bin/env bash

IMAGE_ROOT_TRAIN=../vehicle-triplet-reid/VRIC/train_images ; shift
#IMAGE_ROOT_TEST=../vehicle-triplet-reid/VRIC/gallery_images; shift
INIT_CHECKPT=./pre_trained_model/resnet_v1_50.ckpt;  shift
EXP_ROOT=./experiments/VRIC_MSLM_resNet ; shift

python train_fusion_resnet50.py \
    --train_set ../vehicle-triplet-reid/VRIC/vric_train.txt \
    --model_name resnet_v1_50 \
    --head_name fusion_resnet50 \
    --image_root $IMAGE_ROOT_TRAIN  \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --checkpoint_frequency 1000 \
    --flip_augment \
    --crop_augment \
    --embedding_dim 128 \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 224 --pre_crop_width 224 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 3e-4 \
    --train_iterations 40000 \
    --decay_start_iteration 10000 \
    --weight_decay_factor 0.001 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    "$@"
#    --validation_image_root $IMAGE_ROOT_TEST \
#    --validation_set ../vehicle-triplet-reid/VRIC/vric_gallery.txt \
#    --validation_frequency 2 \