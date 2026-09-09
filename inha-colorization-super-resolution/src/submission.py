import os
import torch
import random
import zipfile
import open_clip
import numpy as np
import pandas as pd
import argparse

from PIL import Image
from tqdm import tqdm

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

def submissions(test_csv_path, input_img_dir):
    # ======================== CLIP 임베딩 추출 및 embed_submission.csv 저장 ========================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_df = pd.read_csv(test_csv_path)
    img_ids = test_df['ID'].tolist()

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_model = clip_model.to(device)

    feat_imgs = []

    print("CLIP 임베딩 추출 및 저장 시작")
    for img_id in tqdm(img_ids):
        input_path = os.path.join(input_img_dir, f"{img_id}.png")
        pil_img = Image.open(input_path).convert('RGB')
        input_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_img = clip_model.encode_image(input_tensor)
            feat_img /= feat_img.norm(dim=-1, keepdim=True)

        feat_img = feat_img.cpu().numpy().reshape(-1)
        feat_imgs.append(feat_img)

    # ✅ 임베딩 CSV 저장
    feat_imgs = np.array(feat_imgs)
    vec_columns = [f'vec_{i}' for i in range(feat_imgs.shape[1])]
    feat_submission = pd.DataFrame(feat_imgs, columns=vec_columns)
    feat_submission.insert(0, 'ID', img_ids)

    embed_csv_path = os.path.join(input_img_dir, 'embed_submission.csv')
    feat_submission.to_csv(embed_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ embed_submission.csv 저장 완료: {embed_csv_path}")

    # ✅ 이미지 폴더 압축
    zip_filename = os.path.basename(input_img_dir.rstrip('/\\')) + ".zip"
    zip_path = os.path.join(os.path.dirname(input_img_dir), zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_img_dir):
            for file in files:
                if not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, input_img_dir)
                    zipf.write(file_path, arcname=arcname)

    print(f"✅ 압축 완료: {zip_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv_path', type=str, required=True)
    parser.add_argument('--input_img_dir', type=str, required=True)
    args = parser.parse_args()
    seed_everything(42)

    submissions(
        test_csv_path=args.test_csv_path,
        input_img_dir=args.input_img_dir
    )
