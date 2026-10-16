import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Sample ConvAutoencoder implementation.
    NOTE: This is a generic example. Your actual trained model must have the
    exact same architecture to successfully load a state_dict saved from that
    model. If your real training script used a different network, replace this
    file with your original `train_autoencoder.py` implementation.
    """
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


if __name__ == "__main__":
    # quick sanity check
    model = ConvAutoencoder(latent_dim=256)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print('input:', x.shape, 'output:', y.shape)
