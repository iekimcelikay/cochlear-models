#!/bin/bash
# High-resolution simulation with auto-continue
# 80 CFs, 512 fibers per type, 2 kHz PSTH sampling

python run_wav_cf_batched.py \
    --num-cf 80 \
    --batch-size 8 \
    --num-anf 512 512 512 \
    --fs-target 2000 \
    --peripheral-fs 100000 \
    --output-dir ./models_output/high_res_sim \
    --experiment-name high_res_80cf \
    --auto-continue