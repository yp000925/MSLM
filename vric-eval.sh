#!/usr/bin/env bash

python ./evaluate.py \
    --excluder diagonal\
    --query_dataset ../vehicle-triplet-reid/VRIC/vric_probe.txt \
    --query_embeddings ./experiments/VRIC_resnet101_test/VRIC_probe_70000_embeddings.h5 \
    --gallery_dataset ../vehicle-triplet-reid/VRIC//vric_gallery.txt \
    --gallery_embeddings ./experiments/VRIC_resnet101_test/VRIC_gallery_70000_embeddings.h5 \
    --metric euclidean\
    --filename ./experiments/VRIC_resnet101_test/VRIC_70000_evaluation.json \
    --batch_size 16

