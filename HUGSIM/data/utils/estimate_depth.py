import argparse
import glob
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from unidepth.models import UniDepthV2
import unidepth.models.backbones.metadinov2.attention as md_attn
from PIL import Image
import json


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_opts()
    
    print('loading depth model...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        # Some xformers builds import successfully but do not have usable CUDA kernels.
        # Force PyTorch SDPA path in UniDepth to avoid runtime NotImplementedError.
        try:
            import xformers.ops as xops
            q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
            _ = xops.memory_efficient_attention(q, q, q)
            print("xformers CUDA attention: available")
        except Exception as e:
            md_attn.XFORMERS_AVAILABLE = False
            print(f"xformers CUDA attention unavailable -> fallback to PyTorch SDPA ({type(e).__name__})")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    model = model.to(device)
    model.eval()
    print("Depth model loaded")
    
    os.makedirs(os.path.join(args.out, 'depth'), exist_ok=True)
    for cam_pth in glob.glob(os.path.join(args.out, 'images', '*')):
        cam = os.path.basename(cam_pth)
        os.makedirs(os.path.join(args.out, 'depth', cam), exist_ok=True)
    
    with open(os.path.join(args.out, 'meta_data.json')) as f:
        meta_data = json.load(f)
    
    for frame in tqdm(meta_data['frames']):
        im_path = os.path.join(args.out, frame['rgb_path'])
        K = np.array(frame['intrinsics'])
        K = torch.from_numpy(K[:3, :3]).float().to(device)
        image = torch.from_numpy(np.array(Image.open(im_path))).permute(2, 0, 1)
        prediction = model.infer(image, K)
        depth = prediction["depth"][0][0].detach().cpu()  # Depth in [m].
        
        depth_path = os.path.join(
            args.out,
            im_path.replace("images", "depth")
            .replace("./", "")
            .replace(".jpg", ".pt")
            .replace(".png", ".pt"),
        )
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)
        
        torch.save(depth, depth_path)
