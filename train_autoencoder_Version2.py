import os
import json
import argparse
import csv
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# 위의 ConvAutoencoder를 import하거나 동일 정의를 사용
from train_autoencoder import ConvAutoencoder

def find_bbox_from_residual(residual_map, min_area=200, threshold_ratio=0.6):
    """
    residual_map: 2D numpy array, 값이 클수록 결함 가능성이 높음
    return: (x1, y1, x2, y2) or None
    """
    # normalize
    r = residual_map.copy()
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)

    # threshold by high anomaly regions
    mask = r > threshold_ratio
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    if (x2 - x1 + 1) * (y2 - y1 + 1) < min_area:
        return None

    return (int(x1), int(y1), int(x2), int(y2))


def infer_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        x = image_tensor.to(device).unsqueeze(0)
        recon = model(x)
        residual = (recon - x).pow(2).squeeze(0).cpu().numpy()
        # channel-wise mean residual map
        residual_map = residual.mean(axis=0)
        mse_score = float(np.mean(residual_map))
    return mse_score, residual_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--threshold-json", type=str, required=True)
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="results.csv")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if __name__ == "__main__" and __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    with open(args.threshold_json, "r") as f:
        meta = json.load(f)
    threshold = meta["threshold"]

    model = ConvAutoencoder(latent_dim=256).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    image_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        image_paths.extend(glob(os.path.join(args.input_folder, f"**/*.{ext}"), recursive=True))

    rows = []
    for p in tqdm(image_paths):
        img = Image.open(p).convert("RGB")
        x = transform(img)
        score, residual_map = infer_image(model, x, device)

        label = "defect" if score > threshold else "normal"
        bbox = None
        if label == "defect":
            bbox = find_bbox_from_residual(residual_map, min_area=50, threshold_ratio=0.7)

        rows.append({
            "path": p,
            "score": float(score),
            "threshold": float(threshold),
            "label": label,
            "bbox": bbox
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "score", "threshold", "label", "bbox"])
        for r in rows:
            bbox_str = ""
            if r["bbox"] is not None:
                bbox_str = str(r["bbox"])
            writer.writerow([r["path"], r["score"], r["threshold"], r["label"], bbox_str])

    print(f"Saved {len(rows)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
