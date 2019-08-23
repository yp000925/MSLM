#!/usr/bin/env bash


python embed.py \
        --experiment_root ./experiments/VRIC_resnet101_test \
        --dataset ../vehicle-triplet-reid/VRIC/vric_probe.txt \
        --filename VRIC_probe_70000_embeddings.h5 \
        --image_root ../vehicle-triplet-reid/VRIC/probe_images \
        --checkpoint checkpoint-70000 \
        --batch_size 128
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean

