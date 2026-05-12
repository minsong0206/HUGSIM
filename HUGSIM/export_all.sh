#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

BASE=/home/ms/HUGSIM_N/HUGSIM
INPUT_BASE=${BASE}/Datasets/recon_data
OUTPUT_BASE=${BASE}/Datasets/export_data

seq_list=("scene-0051" "scene-0411" "scene-0655")

cd ${BASE}

for seq in "${seq_list[@]}"; do
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  Export  Scene: ${seq}"
    echo "████████████████████████████████████████████████████████"

    input_path=${INPUT_BASE}/${seq}
    output_path=${OUTPUT_BASE}/${seq}
    mkdir -p ${output_path}

    echo "▶ export_scene.py — ${seq}"
    python -u eval_render/export_scene.py --model_path ${input_path} \
        --output_path ${output_path} --iteration 30000
    if [[ $? -ne 0 ]]; then
        echo "✖ FAILED: export_scene.py — ${seq}"
        exit 1
    fi
done

    echo "✔ export_scene.py done — ${seq}"