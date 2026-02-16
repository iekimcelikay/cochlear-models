#!/bin/bash
# Resume processing from batch 5 onward

python run_wav_cf_batched.py \
    --num-cf 80 \
    --batch-size 8 \
    --start-from 5 \
    --num-anf 512 512 512 \
    --fs-target 2000 \
    --output-dir ./models_output/high_res_sim \
    --experiment-name high_res_80cf \
    --auto-continue

