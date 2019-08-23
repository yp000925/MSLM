#!/usr/bin/env bash

python ./evaluate.py \
    --excluder veri776 \
    --query_dataset ../vehicle-triplet-reid/VeRi_with_plate/veri_query.txt \
    --query_embeddings ./experiments/VeRi776_resnet101/VeRi_query_70000_embeddings.h5 \
    --gallery_dataset ../vehicle-triplet-reid/VeRi_with_plate/veri_test.txt \
    --gallery_embeddings ./experiments/VeRi776_resnet101/VeRi_test_70000_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/VeRi776_resnet101/VeRi_70000_evaluation.json \
    --batch_size 16 \
