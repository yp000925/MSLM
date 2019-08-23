#!/usr/bin/env bash

python embed.py \
        --experiment_root ./experiments/VeRi776_resnet101 \
        --dataset ../vehicle-triplet-reid/VeRi_with_plate/veri_query.txt \
        --filename Veri_query_70000_embeddings.h5 \
        --image_root ../vehicle-triplet-reid/VeRi_with_plate/image_query \
        --checkpoint checkpoint-70000 \
        --batch_size 128 \
#         --flip_augment \
#         --crop_augment five \
#         --aggregator mean