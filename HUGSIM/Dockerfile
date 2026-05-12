# ==============================================================================
# HUGSIM - NuScenes Preprocessing + Training Environment
# Base: CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
# Python: 3.11 / PyTorch: 2.4.1+cu118
#
# [시행착오 반영 항목]
#   FIX-1  tinycudann: GPU 없는 빌드 환경에서 cc86/cc89 강제 지정
#   FIX-2  xFormers: cu118 호환 버전 0.0.22.post4 사용
#          (0.0.28.post1+cu118 wheel 미존재, cu121 wheel 설치 시 CUDA ops 불일치)
#   FIX-3  UniDepth ops: pip install만으로는 CUDA extension 빌드 안 됨
#          knn / extract_patches compile.sh 별도 실행 필요
#   FIX-4  UniDepth attention.py: XFORMERS_AVAILABLE=False 패치
#          (xFormers CUDA ops가 cu118에서 동작하지 않아 NotImplementedError 발생)
#   FIX-5  simple-knn: HUGSIM submodule에서 직접 빌드
#   FIX-6  PATH: /usr/bin:/bin 항상 포함 (pixi 환경에서 find/wc 등 누락 방지)
# ==============================================================================

FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:/usr/bin:/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# [FIX-1] tinycudann: 빌드 시 GPU 없어도 cc86(A6000) / cc89(RTX40xx) 대상으로 컴파일
ENV TCNN_CUDA_ARCHITECTURES="86;89"
# [FIX-7] gsplat/pytorch3d 등 CUDA extension 빌드 시 arch list 필요
#          GPU 없는 빌드 환경에서 TORCH_CUDA_ARCH_LIST 미설정 시 IndexError 발생
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# ------------------------------------------------------------------------------
# 1. 시스템 패키지
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ninja-build \
    libboost-all-dev \
    libeigen3-dev \
    libfreeimage-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libsqlite3-dev \
    libglew-dev \
    libcgal-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libflann-dev \
    libmetis-dev \
    libceres-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    pkg-config \
    ffmpeg \
    zsh \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python3.11을 기본 python/python3로 등록
# ensurepip 대신 python3-pip 패키지로 pip 설치 (Ubuntu 22.04에서 ensurepip 비활성화됨)
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
 && pip install --upgrade pip setuptools==78.1.0 wheel

# ------------------------------------------------------------------------------
# 2. COLMAP 3.9.1 (소스 빌드)
# ------------------------------------------------------------------------------
RUN git clone https://github.com/colmap/colmap.git /opt/colmap \
 && cd /opt/colmap \
 && git checkout 3.9.1 \
 && mkdir build && cd build \
 && cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DGUI_ENABLED=OFF \
 && ninja -j$(nproc) \
 && ninja install \
 && rm -rf /opt/colmap

# ------------------------------------------------------------------------------
# 3. PyTorch 2.4.1 + cu118
# ------------------------------------------------------------------------------
RUN pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118

# [FIX-2] xFormers 설치 후 torch를 cu118로 재고정
#   - xformers 0.0.22.post4가 torch 2.1.0+cu118을 의존성으로 끌어당겨 torch 버전이 변경됨
#   - 설치 후 torch 2.4.1+cu118로 재설치하여 버전 고정
RUN pip install \
    xformers==0.0.22.post4 \
    --index-url https://download.pytorch.org/whl/cu118 \
 && pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --force-reinstall

# ------------------------------------------------------------------------------
# 4. Python 핵심 의존성
# ------------------------------------------------------------------------------
RUN pip install \
    torchmetrics==1.7.1 \
    roma \
    matplotlib \
    tqdm \
    opencv-python \
    scipy \
    einops \
    trimesh \
    tensorboard \
    gpustat \
    scikit-learn \
    open3d==0.18.0 \
    timm \
    geometry==0.0.23 \
    "gymnasium==1.1.1" \
    imageio \
    jaxtyping \
    mediapy \
    omegaconf \
    pillow \
    plyfile \
    pyquaternion \
    pyyaml \
    shapely \
    tabulate \
    "protobuf==3.20.2" \
    "runx==0.0.6" \
    fire \
    h5py \
    nose \
    dominate \
    cffi \
    piexif \
    jupyter \
    ipdb \
    "numpy>=1.24.3,<2" \
    torch-fidelity \
    huggingface-hub \
    moviepy \
    scikit-image \
    pandas \
    ninja

# ------------------------------------------------------------------------------
# 5. NVIDIA Apex (InverseForm용 — CUDA extension 포함 컴파일)
# ------------------------------------------------------------------------------
RUN git clone https://github.com/NVIDIA/apex.git /opt/apex \
 && cd /opt/apex \
 && git checkout ac8214ee6ba77c0037c693828e39d83654d25720 \
 && sed -i 's/if (bare_metal_version != torch_binary_version):/if False:  # patched: skip cuda version check/' setup.py \
 && python setup.py install --cpp_ext --cuda_ext \
 && rm -rf /opt/apex

# ------------------------------------------------------------------------------
# 6. gsplat (HUGSIM fork)
# ------------------------------------------------------------------------------
RUN pip install --no-build-isolation \
    "git+https://github.com/hyzhou404/HUGSIM_splat.git@main"

# ------------------------------------------------------------------------------
# 7. tinycudann
# [FIX-1] TCNN_CUDA_ARCHITECTURES ENV로 cc86/cc89 강제 지정됨
# ------------------------------------------------------------------------------
RUN pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn@master#subdirectory=bindings/torch"

# ------------------------------------------------------------------------------
# 8. pytorch3d
# ------------------------------------------------------------------------------
RUN pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# ------------------------------------------------------------------------------
# 9. 나머지 git 패키지
# ------------------------------------------------------------------------------
RUN pip install \
    "git+https://github.com/ChristophReich1996/Optical-Flow-Visualization-PyTorch@main" \
    "git+https://github.com/hyzhou404/trajdata.git@main" \
    "git+https://github.com/autonomousvision/kitti360Scripts.git@master" \
    "git+https://github.com/gdlg/simple-waymo-open-dataset-reader.git@master" \
    "git+https://github.com/hyzhou404/nuscenes-devkit.git@master"

# ------------------------------------------------------------------------------
# 10. UniDepth 설치 + attention.py 패치
# [FIX-3] knn/extract_patches CUDA ops 빌드는 실제 GPU 필요 → 컨테이너 실행 후 수동 실행
#         compile.sh 내부에서 torch.cuda.is_available() 체크하므로 빌드 중 GPU 없으면 실패
#
#         컨테이너 실행 후 아래 명령어로 수동 빌드:
#           UNIDEPTH_OPS=$(python -c "import unidepth, os; print(os.path.dirname(unidepth.__file__))")/ops
#           cd ${UNIDEPTH_OPS}/knn && bash compile.sh
#           cd ${UNIDEPTH_OPS}/extract_patches && bash compile.sh
#
# [FIX-4] xFormers CUDA ops가 cu118에서 동작하지 않아
#         memory_efficient_attention_forward NotImplementedError 발생
#         → XFORMERS_AVAILABLE=False 패치로 PyTorch 내장 scaled_dot_product_attention 사용
# ------------------------------------------------------------------------------
RUN pip install "git+https://github.com/hyzhou404/UniDepth.git@main"

RUN find /usr/local/lib/python3.11 -path "*/unidepth/models/backbones/metadinov2/attention.py" \
    -exec sed -i \
        's/XFORMERS_AVAILABLE = XFORMERS_AVAILABLE and torch.cuda.is_available()/XFORMERS_AVAILABLE = False  # patched: xFormers CUDA ops unavailable with cu118/' \
        {} \;

# ------------------------------------------------------------------------------
# 11. Flask (gui/app.py 시뮬레이터 서버용)
# ------------------------------------------------------------------------------
RUN pip install flask

# ------------------------------------------------------------------------------
# 12. HUGSIM 코드베이스 세팅
#
# 이 이미지에는 HUGSIM 코드를 포함하지 않습니다.
# 컨테이너 실행 후 아래 순서로 직접 세팅하세요:
#
#   1) git clone https://github.com/hyzhou404/HUGSIM.git /workspace
#   2) cd /workspace
#   3) pip install --no-build-isolation ./submodules/simple-knn
#   4) pip install -e ./sim
#   5) export PYTHONPATH=/workspace:/workspace/data:$PYTHONPATH
#
# 학습:
#   CUDA_VISIBLE_DEVICES=0 python -u train_ground.py --data_cfg ./configs/nusc.yaml \
#       --source_path <input_path> --model_path <output_path>
#   CUDA_VISIBLE_DEVICES=0 python -u train.py --data_cfg ./configs/nusc.yaml \
#       --source_path <input_path> --model_path <output_path>
#
# 시뮬레이터 GUI 서버 (포트 5000):
#   python gui/app.py --scene <export_path> --car_folder <car_folder>
# ------------------------------------------------------------------------------
WORKDIR /workspace

ENV PYTHONPATH=/workspace:/workspace/data

EXPOSE 5000

# ------------------------------------------------------------------------------
# 기본 진입점
# ------------------------------------------------------------------------------
CMD ["/bin/bash"]
