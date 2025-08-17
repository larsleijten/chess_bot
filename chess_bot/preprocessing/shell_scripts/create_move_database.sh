#!/bin/bash
python /home/larsleijten/repositories/chess_bot/chess_bot/preprocessing/create_move_database.py \
--pgn_zst_file "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/lichess_db_standard_rated_2016-05.pgn.zst" \
--output_dir "/mnt/c/Users/z515232/Documents/hobby_websites/chess_bot/datasets"