# swinir_upscale.py

import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.transforms.functional import to_tensor, to_pil_image
from basicsr.archs.swinir_arch import SwinIR

def seed_everything(seed: int = 42):
    print(f"✅ Setting seed: {seed}")
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)                # CPU
    torch.cuda.manual_seed(seed)           # Current GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs (if multi-GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_swinir_model():
    weight_path = './SwinIR/weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth'
    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict['params'], strict=True)
    model.eval().cuda()

    return model

def upscale_with_swinir(pil_img, model):
    lr = to_tensor(pil_img).unsqueeze(0).cuda()  # (1, 3, H, W)
    with torch.no_grad():
        sr = model(lr)
    sr = sr.clamp(0, 1)

    return to_pil_image(sr.squeeze(0).cpu())

def upscale_folder(input_dir, output_dir, model):
    os.makedirs(output_dir, exist_ok=True)
    img_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in tqdm(img_list, desc=f"Upscaling {input_dir} -> {output_dir}"):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        sr_img = upscale_with_swinir(img, model)
        sr_img.save(os.path.join(output_dir, img_name))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(42)

    model = load_swinir_model()
    upscale_folder(args.input_dir, args.output_dir, model)
