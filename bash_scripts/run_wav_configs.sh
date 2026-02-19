#!/bin/bash
# Master script to choose configuration

echo "Select configuration:"
echo "1) High resolution (80 CFs, auto-continue)"
echo "2) Test (2 batches with pauses)"
echo "3) Moderate resolution (50 CFs)"
echo "4) Custom CF range (100 CFs, 200-5000 Hz)"
echo "5) Resume from batch 5"
read -p "Enter choice [1-5]: " choice

case $choice in
    1) ./run_high_res_auto.sh ;;
    2) ./run_test_2batches.sh ;;
    3) ./run_moderate_resolution.sh ;;
    4) ./run_custom_cfs.sh ;;
    5) ./resume_from_batch5.sh ;;
    *) echo "Invalid choice" ;;
esac