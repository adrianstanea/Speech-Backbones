#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

checkpoint_path='/workspace/local/Speech-Backbones/Grad-TTS/logs/no_ASCII_conversion/grad_1000.pt'
n_epochs=100
save_every=5

train_filelists=(
    '/workspace/local/Matcha-TTS/resources/filelists/SWARA_1_0/finetune_meta_bas_10.csv'
    '/workspace/local/Matcha-TTS/resources/filelists/SWARA_1_0/finetune_meta_bas_950.csv'
    '/workspace/local/Matcha-TTS/resources/filelists/SWARA_1_0/finetune_meta_sgs_10.csv'
    '/workspace/local/Matcha-TTS/resources/filelists/SWARA_1_0/finetune_meta_sgs_950.csv'
)

current_date=$(date +%Y%m%d)

for train_filelist in "${train_filelists[@]}"; do
    # echo "Processing filelist: $train_filelist"

    # Extract the filename from the full path
    filename=$(basename "$train_filelist")

    # Remove the prefix "finetune_meta_" and suffix ".csv"
    base_identifier="${filename#finetune_meta_}"
    base_identifier="${base_identifier%.csv}"

    # Split the remaining string (e.g., "bas_10") by underscore
    # and directly concatenate the parts
    speaker_part=$(echo "$base_identifier" | cut -d'_' -f1)
    number_part=$(echo "$base_identifier" | cut -d'_' -f2)

    if [ "$number_part" -eq 10 ]; then
        batch_size=5
    else
        batch_size=16
    fi

    combined_identifier="${speaker_part}_${number_part}"

    # Append the current date
    log_dir="log_dir/${combined_identifier}_${current_date}"

    echo "Extracted and dated string: $log_dir"

    python3 train.py \
        --checkpoint_path "$checkpoint_path" \
        --n_epochs "$n_epochs" \
        --log_dir "$log_dir" \
        --save_every "$save_every" \
        --train_filelist_path "$train_filelist" \
        --batch_size "$batch_size"
done