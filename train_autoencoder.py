import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for fixed-size RGB images."""
    def __init__(self, latent_dim=256):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(True),
        )
        self._enc_out_shape = (256, 16, 16)
        enc_flat = 256 * 16 * 16
        self.fc_mu = nn.Linear(enc_flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, enc_flat)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 256x256
            nn.Sigmoid(),
        )

    def forward(self, x):
        b = x.size(0)
        z = self.encoder(x)
        z = z.view(b, -1)
        z = self.fc_mu(z)
        z = self.fc_dec(z)
        z = z.view(b, *self._enc_out_shape)
        x_hat = self.decoder(z)
        return x_hat


class ImageFolderDataset(Dataset):
    """Loads images under a folder; class subfolders are filtered if requested."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, paths, img_size=256):
        self.paths = list(paths)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def collect_images(root, class_names=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ImageFolderDataset.extensions)
    if class_names:
        paths = [p for p in paths if any(part.lower() in class_names for part in p.relative_to(root).parts[:-1])]
    return paths


def reconstruction_scores(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores.extend((model(batch) - batch).pow(2).mean(dim=(1, 2, 3)).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def train(args):
    if args.img_size != 256:
        raise ValueError("This model architecture requires --img-size 256.")
    if not 0 < args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    root = Path(args.data_dir)

    # archive/intact/{top,side} and archive/damaged/{top,side} are supported.
    intact_names = {"intact", "normal", "normals"}
    damaged_names = {"damaged", "defect", "defects", "bad"}
    all_paths = collect_images(root)
    intact_paths = [p for p in all_paths if any(part.lower() in intact_names for part in p.relative_to(root).parts[:-1])]
    if not intact_paths:
        # Also accept a root containing only normal images (legacy data layout).
        intact_paths = [p for p in all_paths if not any(part.lower() in damaged_names for part in p.relative_to(root).parts[:-1])]
    if len(intact_paths) < 2:
        raise ValueError(f"Need at least 2 intact training images under {root}; found {len(intact_paths)}")

    full_ds = ImageFolderDataset(intact_paths, args.img_size)
    val_size = max(1, int(len(full_ds) * args.validation_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ConvAutoencoder(args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {total / train_size:.6f}")

    intact_scores = reconstruction_scores(model, val_loader, device)
    damaged_paths = [p for p in all_paths if any(part.lower() in damaged_names for part in p.relative_to(root).parts[:-1])]
    damaged_scores = []
    if damaged_paths:
        damaged_loader = DataLoader(ImageFolderDataset(damaged_paths, args.img_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        damaged_scores = reconstruction_scores(model, damaged_loader, device).tolist()

    # If labeled defect examples are present, calibrate threshold for balanced accuracy.
    if damaged_scores:
        candidates = np.unique(np.concatenate([intact_scores, damaged_scores]))
        best = max(candidates, key=lambda threshold: (
            (np.mean(intact_scores <= threshold) + np.mean(np.asarray(damaged_scores) > threshold)) / 2,
            -threshold,
        ))
        threshold = float(best)
        threshold_method = "balanced_accuracy"
    else:
        threshold = float(np.percentile(intact_scores, args.percentile))
        threshold_method = f"intact_validation_percentile_{args.percentile:g}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out_dir / args.model_file)
    metadata = {
        "threshold": threshold, "threshold_method": threshold_method,
        "img_size": args.img_size, "latent_dim": args.latent_dim,
        "intact_train_images": train_size, "intact_validation_images": val_size,
        "damaged_calibration_images": len(damaged_scores),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "threshold.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model and threshold ({threshold:.6f}, {threshold_method}) to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an autoencoder using intact images.")
    parser.add_argument("--data-dir", default="archive")
    parser.add_argument("--out-dir", default="model_out")
    parser.add_argument("--model-file", default="ae_final.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    train(parser.parse_args())
