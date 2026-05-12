#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

BASE=/home/ms/HUGSIM_N/HUGSIM
INPUT_BASE=${BASE}/Datasets/export_data
OUTPUT_BASE=${BASE}/Datasets/render_data

seq_list=("scene-0051" "scene-0411" "scene-0655")

cd ${BASE}

for seq in "${seq_list[@]}"; do
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  Rendering   Scene: ${seq}"
    echo "████████████████████████████████████████████████████████"

    input_path=${INPUT_BASE}/${seq}
    output_path=${OUTPUT_BASE}/${seq}
    mkdir -p ${output_path}

    echo "▶ render_scene.py — ${seq}"
    python -u eval_render/render.py --model_path ${input_path} \
        --iteration 30000
    if [[ $? -ne 0 ]]; then
        echo "✖ FAILED: render_scene.py — ${seq}"
        exit 1
    fi
    echo "✔ render_scene.py done — ${seq}"
done