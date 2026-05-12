# #!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

BASE=/home/ms/HUGSIM_N/HUGSIM
INPUT_BASE=${BASE}/Datasets/asap_data
OUTPUT_BASE=${BASE}/Datasets/recon_split_data

seq_list=("scene-0051" "scene-0411" "scene-0655")

cd ${BASE}

for seq in "${seq_list[@]}"; do
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  Training: ${seq}"
    echo "████████████████████████████████████████████████████████"

    input_path=${INPUT_BASE}/${seq}
    output_path=${OUTPUT_BASE}/${seq}
    mkdir -p ${output_path}

    echo "▶ [1/2] train_ground.py — ${seq}"
    python -u train_ground.py --data_cfg ./configs/nusc.yaml \
        --source_path ${input_path} --model_path ${output_path}
    if [[ $? -ne 0 ]]; then
        echo "✖ FAILED: train_ground.py — ${seq}"
        exit 1
    fi
    echo "✔ train_ground.py done — ${seq}"

    echo "▶ [2/2] train.py — ${seq}"
    python -u train.py --data_cfg ./configs/nusc.yaml \
        --source_path ${input_path} --model_path ${output_path}
    if [[ $? -ne 0 ]]; then
        echo "✖ FAILED: train.py — ${seq}"
        exit 1
    fi
    echo "✔ train.py done — ${seq}"

    echo ""
    echo "  DONE: ${seq}"
    echo "████████████████████████████████████████████████████████"
done

echo ""
echo "All scenes trained successfully."
