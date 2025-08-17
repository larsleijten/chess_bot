#!/bin/bash

python /home/larsleijten/repositories/chess_bot/chess_bot/train/train.py \
    --data_dir "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/datasets/black" \
    --checkpoint_path "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/train/unet_black.pth" \
    --epochs 10 \
    --batch_size 32