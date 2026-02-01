#!/bin/bash

#   conda activate py_3_6_9_GradTTS

#Store all the csv files from the /workspace/local/transcriptions/redo/grad-tts in a variable list and store the full path to each file
# pt_files=$(find /workspace/local/Speech-Backbones/Grad-TTS/logs/redo -type f -name "*.pt")
pt_files=$(find /workspace/local/checkpoints/grad-tts -type f -name "grad-tts*.pt")

for pt_file in $pt_files; do
    # Get the base name of the file (without the path)
    base_name=$(basename "$pt_file")
    echo "Base name: $base_name"
    echo "Pt file: $pt_file"

    # Get the name of the file without the .pt extension
    file_name="${base_name%.pt}"

    # file names have the form: grad-tts-sgs10-15 extract the speaker pattern where in this case is sgs
    speaker=$(echo "$file_name" | cut -d'-' -f3 | sed 's/[0-9]*//g')
    if [ "$speaker" == "base" ]; then
        speaker="bas"
    fi

    echo "Processing file: $pt_file"
    echo "Base name: $base_name"
    echo "File name: $file_name"
    echo "Speaker pattern: $speaker"

    python3 inference_RO.py \
        --file /workspace/local/evaluation/eval_${speaker}_42.txt \
        -c $pt_file
done
