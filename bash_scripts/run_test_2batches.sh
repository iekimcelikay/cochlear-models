#!/bin/bash
# Test run: first 2 batches only with pauses

python run_wav_cf_batched.py \
    --num-cf 40 \
    --batch-size 20 \
    --num-batches 2 \
    --num-anf 128 128 128 \
    --fs-target 2000 \
    --output-dir ./models_output/test_medium_res \
    --experiment-name test_40cf_dipc
