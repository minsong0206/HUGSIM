#!/bin/zsh

# Ensure standard system tools are reachable regardless of conda/pixi PATH
export PATH="/usr/bin:/bin:${PATH}"
export PYTHONPATH="${PWD}:$PYTHONPATH"

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

# Print a clearly visible step banner
step() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  STEP: $*"
    echo "════════════════════════════════════════════════════════"
}

# Run a Python script and abort immediately on failure.
# Usage: run_py <description> <script> [args...]
run_py() {
    local desc="$1"; shift
    echo "▶ [Python] ${desc}"
    echo "  CMD: python $*"
    python "$@"
    local code=$?
    if [[ $code -ne 0 ]]; then
        echo ""
        echo "✖ FAILED: ${desc}"
        echo "  Exit code: ${code}"
        echo "  CMD: python $*"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit ${code}
    fi
    echo "✔ OK: ${desc}"
}

# Assert a file exists and is non-empty; abort otherwise.
# Usage: assert_file <path> [description]
assert_file() {
    local path="$1"
    local desc="${2:-${path}}"
    if [[ ! -f "${path}" ]]; then
        echo ""
        echo "✖ VALIDATION FAILED: expected file not found"
        echo "  File: ${path}"
        echo "  Step: ${desc}"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit 1
    fi
    if [[ ! -s "${path}" ]]; then
        echo ""
        echo "✖ VALIDATION FAILED: file exists but is empty"
        echo "  File: ${path}"
        echo "  Step: ${desc}"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit 1
    fi
    echo "  ✔ exists & non-empty: ${path}"
}

# Assert a directory exists and contains at least one file.
assert_dir() {
    local path="$1"
    local desc="${2:-${path}}"
    if [[ ! -d "${path}" ]]; then
        echo ""
        echo "✖ VALIDATION FAILED: expected directory not found"
        echo "  Dir: ${path}"
        echo "  Step: ${desc}"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit 1
    fi
    # Use zsh glob: (N) suppresses errors, (.) matches regular files only
    local -a files; files=( ${path}/*(N) )
    local count=${#files}
    if [[ ${count} -eq 0 ]]; then
        echo ""
        echo "✖ VALIDATION FAILED: directory exists but contains no entries"
        echo "  Dir: ${path}"
        echo "  Step: ${desc}"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit 1
    fi
    echo "  ✔ directory OK (${count} entries): ${path}"
}

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────

cuda=0
data='/home/ms/HUGSIM/WD_BLACK_4TB/HUGSIM/Nuscenes_RAW_DATA/trainval'
version='interp_12Hz_trainval'

# seq_list=('scene-0411' 'scene-0064' 'scene-0038' 'scene-0013')
seq_list=('scene-0038')

# ─────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────

for seq in "${seq_list[@]}"; do
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  Processing: ${seq}"
    echo "████████████████████████████████████████████████████████"

    start=0
    end=180
    out=/home/ms/WD_BLACK_4TB/HUGSIM/Data/asap_data/${seq}

    export CUDA_VISIBLE_DEVICES=$cuda
    mkdir -p ${out}

    # ── 1. Load NuScenes data ────────────────────────────────
    step "1/9  nusc/load.py  →  ${seq}"
    run_py "Load NuScenes scene" \
        nusc/load.py --datapath ${data} --version ${version} --seq ${seq} --out ${out} \
        --start ${start} --end ${end} --downsample 2 --video

    assert_file "${out}/meta_data.json"    "nusc/load.py output"
    assert_dir  "${out}/images"            "nusc/load.py output"

    # ── 2. Visualise 2D bounding boxes (pre-mask) ───────────
    step "2/9  utils/vis_bbox_2d.py  (pre-mask)"
    run_py "Visualise 2D bboxes (pre-mask)" \
        utils/vis_bbox_2d.py --out ${out}

    # ── 3. Semantic mask (InverseForm) ───────────────────────
    step "3/9  InverseForm/infer_nuscenes.sh"
    cd InverseForm
    ./infer_nuscenes.sh ${cuda} ${out}
    inverseform_code=$?
    cd -
    if [[ ${inverseform_code} -ne 0 ]]; then
        echo "✖ FAILED: InverseForm infer_nuscenes.sh  (exit ${inverseform_code})"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit ${inverseform_code}
    fi
    echo "✔ OK: InverseForm"
    assert_dir "${out}/semantics" "InverseForm output"

    # ── 4. Dynamic mask ─────────────────────────────────────
    step "4/9  utils/create_dynamic_mask.py"
    run_py "Create dynamic mask" \
        utils/create_dynamic_mask.py --data_path ${out} --data_type nuscenes

    assert_dir "${out}/masks" "create_dynamic_mask.py output"

    # ── 5. COLMAP sparse reconstruction ─────────────────────
    step "5/9  COLMAP  (prepare + triangulate + BA)"
    rm -rf ${out}/colmap_sparse*
    rm -f  ${out}/database.db*
    rm -rf ${out}/prior

    run_py "Prepare COLMAP (feature extract + triangulate + rigid BA)" \
        nusc/prepare_colmap.py -i ${out}

    # validate all three COLMAP outputs
    assert_file "${out}/colmap_sparse_tri/cameras.bin"  "prepare_colmap.py → colmap_sparse_tri"
    assert_file "${out}/colmap_sparse_tri/images.bin"   "prepare_colmap.py → colmap_sparse_tri"
    assert_file "${out}/colmap_sparse_tri/points3D.bin" "prepare_colmap.py → colmap_sparse_tri"
    assert_file "${out}/colmap_sparse_ba/cameras.bin"   "prepare_colmap.py → colmap_sparse_ba"
    assert_file "${out}/colmap_sparse_ba/images.bin"    "prepare_colmap.py → colmap_sparse_ba"
    assert_file "${out}/colmap_sparse_ba/points3D.bin"  "prepare_colmap.py → colmap_sparse_ba"

    # ── 6. Convert sparse model to PLY ──────────────────────
    step "6/9  colmap model_converter  →  sparse_ba.ply"
    echo "▶ [colmap] model_converter"
    colmap model_converter \
        --input_path ${out}/colmap_sparse_tri \
        --output_path ${out}/sparse_ba.ply \
        --output_type PLY
    colmap_converter_code=$?
    if [[ ${colmap_converter_code} -ne 0 ]]; then
        echo "✖ FAILED: colmap model_converter  (exit ${colmap_converter_code})"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit ${colmap_converter_code}
    fi
    assert_file "${out}/sparse_ba.ply" "colmap model_converter output"
    echo "✔ OK: colmap model_converter"

    # ── 7. Update camera poses from COLMAP BA ───────────────
    step "7/9  colmap/update_campose.py"
    run_py "Update camera poses" \
        colmap/update_campose.py --datapath ${out}

    assert_file "${out}/meta_data.json"      "update_campose.py"
    assert_file "${out}/meta_data_init.json" "update_campose.py  (backup)"

    # ── 8. Re-visualise 2D bboxes (post-update) ─────────────
    step "8/9  utils/vis_bbox_2d.py  (post-campose)"
    run_py "Visualise 2D bboxes (post-update)" \
        utils/vis_bbox_2d.py --out ${out}

    # ── 9. Depth estimation + merging ───────────────────────
    step "9a/9  utils/estimate_depth.py"
    run_py "Estimate depth" \
        utils/estimate_depth.py --out ${out}

    assert_dir "${out}/depth"                "estimate_depth.py output (depth/ dir)"
    # depth files are inside per-camera subdirs — check at least one .pt file exists
    local -a depth_files; depth_files=( ${out}/depth/**/*.pt(N) )
    if [[ ${#depth_files} -eq 0 ]]; then
        echo ""
        echo "✖ VALIDATION FAILED: no .pt depth files found under ${out}/depth/"
        echo "  Step: estimate_depth.py output"
        echo "  Scene: ${seq}  |  Out: ${out}"
        exit 1
    fi
    echo "  ✔ depth files OK (${#depth_files} .pt files found)"

    step "9b/9  utils/merge_depth_wo_ground.py"
    run_py "Merge depth (no ground)" \
        utils/merge_depth_wo_ground.py --out ${out} --total 200000

    assert_file "${out}/points3d.ply" "merge_depth_wo_ground.py output"

    step "9c/9  utils/merge_depth_ground.py"
    run_py "Merge depth (with ground)" \
        utils/merge_depth_ground.py --out ${out} --total 200000 --datatype nuscenes

    assert_file "${out}/ground_points3d.ply" "merge_depth_ground.py output"
    assert_file "${out}/ground_param.pkl"    "merge_depth_ground.py output"

    # ── Done ─────────────────────────────────────────────────
    echo ""
    echo "████████████████████████████████████████████████████████"
    echo "  DONE: ${seq}"
    echo "████████████████████████████████████████████████████████"
done

echo ""
echo "All scenes processed successfully."