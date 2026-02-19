#!/bin/bash
# Custom CF range and multiple WAV files

python run_wav_cf_batched.py \
    --num-cf 100 \
    --min-cf 200 \
    --max-cf 5000 \
    --batch-size 10 \
    --num-anf 128 128 128 \
    --fs-target 500 \
    --wav-files ./stimuli/produced/*.wav \
    --output-dir ./models_output/custom_range \
    --experiment-name custom_100cf \
    --auto-continue