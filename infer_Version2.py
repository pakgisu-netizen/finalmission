#!/usr/bin/env python3
"""
Inference: compute reconstruction MSE for images in a folder and decide normal/defect.
Usage:
  python infer.py --model-dir model_out --input-folder new_images --out-csv results.csv
"""
import argparse
import os
import json
from glob import glob
import csv
from tqdm import tqdm

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from train_autoencoder import ConvAutoencoder, ImageFolderDataset

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    meta_path = os.path.join(args.model_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError('metadata.json not found in model_dir')
    meta = json.load(open(meta_path))
    thr = meta['threshold']
    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    model_path = os.path.join(args.model_dir, args.model_file)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((meta.get('img_size', args.img_size), meta.get('img_size', args.img_size))),
        transforms.ToTensor(),
    ])

    files = []
    for ext in ('jpg','jpeg','png'):
        files += glob(os.path.join(args.input_folder, f'**/*.{ext}'), recursive=True)
    results = []
    with torch.no_grad():
        for p in tqdm(files):
            img = Image.open(p).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            recon = model(x)
            mse = float(torch.mean((recon - x)**2).cpu().numpy())
            label = 'defect' if mse > thr else 'normal'
            results.append((p, mse, label))
    # save csv
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','mse','label','threshold'])
        for r in results:
            w.writerow([r[0], r[1], r[2], thr])
    print(f'Wrote {len(results)} results to {args.out_csv}. Threshold={thr:.6f}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', required=True)
    p.add_argument('--model-file', default='ae_final.pth')
    p.add_argument('--input-folder', required=True)
    p.add_argument('--out-csv', default='inference_results.csv')
    p.add_argument('--img-size', type=int, default=256)
    p.add_argument('--latent-dim', type=int, default=256)
    p.add_argument('--no-cuda', action='store_true')
    args = p.parse_args()
    infer(args)