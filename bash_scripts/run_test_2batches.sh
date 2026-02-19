#!/bin/bash
# Test run: first 2 batches only with pauses

python run_wav_cf_batched.py \
    --num-cf 80 \
    --batch-size 8 \
    --num-batches 2 \
    --num-anf 512 512 512 \
    --fs-target 2000 \
    --output-dir ./models_output/test_high_res \
    --experiment-name test_80cf