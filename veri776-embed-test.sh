#!/usr/bin/env bash

python embed.py \
        --experiment_root ./experiments/VeRi776_resnet101 \
        --dataset ../vehicle-triplet-reid/VeRi_with_plate/veri_test.txt \
        --filename Veri_test_70000_embeddings.h5 \
        --image_root ../vehicle-triplet-reid/VeRi_with_plate/image_test \
        --checkpoint checkpoint-70000 \
        --batch_size 128 \
#        --flip_augment \
#        --crop_augment five \
#        --aggregator mean

