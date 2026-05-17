"""
Convolutional Variational Autoencoder (ConvVAE).

Architecture at a glance
------------------------
Encoder
  Input (B,3,H,W)
  → [Conv(k=4,s=2,p=1) + BN + ReLU] × num_layers   # halves H,W each step
  → flatten → Linear → mu (B, latent_dim)
                     → logvar (B, latent_dim)

Reparameterisation
  z = mu + exp(0.5 * logvar) * eps,  eps ~ N(0, I)

Decoder
  z → Linear → reshape (B, C, 4, 4)
  → [ConvTranspose(k=4,s=2,p=1) + BN + ReLU] × (num_layers-1)
  → ConvTranspose → Sigmoid     # output (B,3,H,W) in [0,1]

num_layers is derived automatically: we keep halving until 4×4,
so num_layers = log2(image_size) - 2  (e.g. 64→4 needs 4 layers).
base_channels controls model size: 32 for the full model, 16 for lite.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    """Convolutional VAE.

    Args:
        image_size:    Square input image side length (power of 2, ≥ 8).
        latent_dim:    Dimension of the latent vector z.
        base_channels: Channels in the first encoder block; doubled each
                       subsequent block.  Full model: 32, lite model: 16.
    """

    def __init__(
        self, image_size: int, latent_dim: int, base_channels: int
    ) -> None:
        super().__init__()

        assert (image_size >= 8) and (image_size & (image_size - 1)) == 0, (
            f"image_size must be a power of 2 and ≥ 8, got {image_size}"
        )

        # Number of Conv / ConvTranspose blocks.
        # Each block halves (or doubles) the spatial dimension, so we need
        # log2(image_size / 4) steps to go from image_size down to 4×4.
        # image_size=64:  log2(64) -2 = 4 layers, channels 32→64→128→256
        # image_size=256: log2(256)-2 = 6 layers, channels 32→64→128→256→512→1024
        num_layers: int = int(math.log2(image_size)) - 2

        # ------------------------------------------------------------------ #
        # Encoder                                                             #
        # ------------------------------------------------------------------ #
        # Conv(kernel=4, stride=2, padding=1) halves H and W exactly:
        #   out = floor((in + 2*1 - 4) / 2) + 1 = in/2
        # Channel count: 3 → base_ch → 2*base_ch → 4*base_ch → …
        encoder_blocks: list[nn.Module] = []
        in_ch = 3
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)   # 32, 64, 128, 256 …
            encoder_blocks += [
                # bias=False: BatchNorm's learnable beta covers the bias
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.encoder = nn.Sequential(*encoder_blocks)

        # After all encoder blocks the spatial size is always 4×4.
        self.feature_channels: int = in_ch   # = base_channels * 2^(num_layers-1)
        self.feature_spatial: int = 4
        flat_dim: int = self.feature_channels * self.feature_spatial ** 2

        # Two linear heads output the parameters of q(z | x) = N(mu, diag(exp(logvar))).
        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # ------------------------------------------------------------------ #
        # Decoder                                                             #
        # ------------------------------------------------------------------ #
        # A linear layer maps z back to the flat bottleneck, then we reshape
        # and mirror the encoder with ConvTranspose blocks.
        # ConvTranspose(kernel=4, stride=2, padding=1) doubles H and W:
        #   out = (in - 1)*2 - 2*1 + 4 = 2*in
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        decoder_blocks: list[nn.Module] = []
        dec_in_ch = self.feature_channels       # start from the bottleneck channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            dec_out_ch = 3 if is_last else dec_in_ch // 2

            decoder_blocks.append(
                nn.ConvTranspose2d(
                    dec_in_ch, dec_out_ch, kernel_size=4, stride=2, padding=1,
                    # Keep bias on the final layer (no BN follows it)
                    bias=is_last,
                )
            )
            if is_last:
                # Sigmoid maps pixel values to [0, 1] to match the input
                # range — no mean/std normalisation is applied to the data.
                decoder_blocks.append(nn.Sigmoid())
            else:
                decoder_blocks += [
                    nn.BatchNorm2d(dec_out_ch),
                    nn.ReLU(inplace=True),
                ]
            dec_in_ch = dec_out_ch

        self.decoder = nn.Sequential(*decoder_blocks)

        # Exposed so that sample() and callers can reference them directly.
        self.latent_dim = latent_dim

    # ------------------------------------------------------------------ #
    # Building blocks                                                      #
    # ------------------------------------------------------------------ #

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images to the parameters of q(z | x).

        Args:
            x: Input images (B, 3, H, W), values in [0, 1].

        Returns:
            mu:     Mean of the approximate posterior (B, latent_dim).
            logvar: Log-variance of the approximate posterior (B, latent_dim).
        """
        h = self.encoder(x)                 # (B, feature_channels, 4, 4)
        h = h.flatten(start_dim=1)          # (B, feature_channels * 16)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z via the reparameterisation trick.

        Naive sampling z ~ N(mu, sigma^2) would stop gradients from flowing
        back through z into mu and logvar.  The trick rewrites the sample as
            z = mu + sigma * eps,  eps ~ N(0, I)
        so gradients w.r.t. mu and logvar pass through the deterministic
        operations + and *, while only eps is stochastic.

        Args:
            mu:     Mean tensor (B, latent_dim).
            logvar: Log-variance tensor (B, latent_dim).

        Returns:
            z: Sampled latent vector (B, latent_dim).
        """
        std = torch.exp(0.5 * logvar)   # sigma  =  exp(logvar / 2)
        eps = torch.randn_like(std)     # eps ~ N(0, I), identical shape
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector to a reconstructed image.

        Args:
            z: Latent tensor (B, latent_dim).

        Returns:
            x_recon: Reconstructed images (B, 3, H, W), values in [0, 1].
        """
        h = self.fc_decode(z)
        # Reshape the flat vector back to a spatial feature map (B, C, 4, 4)
        h = h.view(
            h.size(0),
            self.feature_channels,
            self.feature_spatial,
            self.feature_spatial,
        )
        return self.decoder(h)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass: encode → reparameterise → decode.

        Args:
            x: Input images (B, 3, H, W), values in [0, 1].

        Returns:
            x_recon: Reconstructed images (B, 3, H, W), values in [0, 1].
            mu:      Posterior mean (B, latent_dim).
            logvar:  Posterior log-variance (B, latent_dim).
            All three are needed to compute the ELBO loss.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(
        self, n: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Generate images by sampling z from the prior p(z) = N(0, I).

        This is the "generation task": we do not need any input image.

        Args:
            n:      Number of images to generate.
            device: Target device; defaults to the model's current device.

        Returns:
            Generated images (n, 3, H, W), values in [0, 1].
        """
        if device is None:
            device = next(self.parameters()).device
        # Draw n random latent vectors from the standard normal prior
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def build_model(cfg: SimpleNamespace) -> ConvVAE:
    """Construct a ConvVAE from a config namespace.

    Args:
        cfg: Config with fields ``image_size``, ``latent_dim``,
             ``base_channels``.

    Returns:
        Initialised ConvVAE on CPU.  Move to GPU after calling this.
    """
    return ConvVAE(
        image_size=cfg.image_size,
        latent_dim=cfg.latent_dim,
        base_channels=cfg.base_channels,
    )
