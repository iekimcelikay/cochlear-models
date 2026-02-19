#!/bin/bash
# Moderate resolution: 50 CFs, 256 fibers, 1 kHz PSTH
# Good balance between resolution and memory

python run_wav_cf_batched.py \
    --num-cf 50 \
    --batch-size 10 \
    --num-anf 256 256 256 \
    --fs-target 1000 \
    --output-dir ./models_output/moderate_res \
    --experiment-name moderate_50cf \
    --auto-continue