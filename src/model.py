"""
GAN model components for the image inpainting project.

Two classes are defined here:

Generator  (Context-Encoder style)
    A deterministic encoder–bottleneck–decoder network.  Input: 4 channels
    (masked RGB + binary mask).  Output: a full 3-channel image prediction
    in [0, 1].  No probabilistic parts (no mu / logvar / reparameterisation)
    — the "generative" quality comes entirely from the adversarial training
    signal, not from a learned prior.

Discriminator  (PatchGAN)
    A fully-convolutional classifier that outputs a 2-D grid of patch-level
    logit scores.  Each score asks "is this receptive-field patch real or
    fake?".  PatchGAN is cheaper than a full-image discriminator and is known
    to drive sharper, better-textured inpainted regions.

Architecture note
-----------------
Both models scale together via ``base_channels``:
  full model  →  base_channels = 32
  lite model  →  base_channels = 16

The Generator encoder and decoder reuse the same Conv/ConvTranspose blocks
as the old ConvVAE (k=4, s=2, p=1) with three changes:
  1. Input is 4 channels (masked_rgb ‖ mask) instead of 3.
  2. The bottleneck is two *deterministic* linear layers — no sampling.
  3. No skip connections between encoder and decoder.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
# Generator                                                            #
# ------------------------------------------------------------------ #

class Generator(nn.Module):
    """Context-Encoder style inpainting generator.

    Receives a 4-channel input ``[masked_rgb (3ch) | mask (1ch)]`` and
    predicts a full 3-channel image.  Call :meth:`complete` to composite
    the prediction with the original known pixels.

    Args:
        image_size:     Square image side length (power of 2, ≥ 8).
        bottleneck_dim: Size of the deterministic bottleneck vector.
                        Replaces the stochastic z of a VAE — this model
                        is fully deterministic given its input.
        base_channels:  Channel count in the first encoder block;
                        doubled at each subsequent block.
                        Full model: 32 → 64 → … → 1024 (6 layers @ 256 px).
                        Lite model: 16 → 32 → … →  512 (6 layers @ 256 px).
    """

    def __init__(
        self,
        image_size: int,
        bottleneck_dim: int,
        base_channels: int,
    ) -> None:
        super().__init__()

        assert (image_size >= 8) and (image_size & (image_size - 1)) == 0, (
            f"image_size must be a power of 2 and ≥ 8, got {image_size}"
        )

        # Number of down/up-sampling blocks.
        # Each Conv halves H and W, so to go from image_size down to 4×4
        # we need log2(image_size / 4) = log2(image_size) - 2 steps.
        # Example: image_size=256 → 6 layers (256→128→64→32→16→8→4).
        num_layers: int = int(math.log2(image_size)) - 2

        # ---- Encoder -------------------------------------------------
        # Conv(kernel=4, stride=2, padding=1) halves each spatial dim:
        #   out = floor((in + 2*pad - k) / s) + 1 = in / 2
        # Input starts at 4 channels (3 masked-RGB + 1 mask).
        # Channels double each layer: base_ch → 2×base_ch → … → 2^(n-1)×base_ch.
        encoder_blocks: list[nn.Module] = []
        in_ch = 4  # 3 RGB + 1 mask channel
        for i in range(num_layers):
            out_ch = base_channels * (2 ** i)  # 32, 64, 128, 256, 512, 1024 …
            encoder_blocks += [
                # bias=False: BatchNorm's learnable β already shifts activations.
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.encoder = nn.Sequential(*encoder_blocks)

        # After all encoder blocks the spatial size is always 4×4.
        self.feature_channels: int = in_ch   # = base_ch * 2^(num_layers-1)
        self.feature_spatial: int = 4
        flat_dim: int = self.feature_channels * self.feature_spatial ** 2

        # ---- Deterministic bottleneck --------------------------------
        # Unlike a VAE we do NOT sample.  We compress the flattened features
        # to bottleneck_dim and expand back — a plain information bottleneck.
        # This still forces the encoder to build a compact representation
        # but produces a fixed output for every input (no noise).
        self.fc_encode = nn.Linear(flat_dim, bottleneck_dim)
        self.fc_decode_up = nn.Linear(bottleneck_dim, flat_dim)

        # ---- Decoder -------------------------------------------------
        # Mirrors the encoder: ConvTranspose(k=4, s=2, p=1) doubles each dim:
        #   out = (in - 1)*s - 2*pad + k = 2*in
        # Channels halve each layer; final layer outputs 3 + Sigmoid.
        # No skip connections — intentional; adding them would make this
        # a U-Net, which is architecturally distinct.
        decoder_blocks: list[nn.Module] = []
        dec_in_ch = self.feature_channels
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            dec_out_ch = 3 if is_last else dec_in_ch // 2

            decoder_blocks.append(
                nn.ConvTranspose2d(
                    dec_in_ch, dec_out_ch,
                    kernel_size=4, stride=2, padding=1,
                    bias=is_last,  # keep bias on the final layer (no BN after it)
                )
            )
            if is_last:
                # Sigmoid constrains output to [0, 1] to match image range.
                decoder_blocks.append(nn.Sigmoid())
            else:
                decoder_blocks += [
                    nn.BatchNorm2d(dec_out_ch),
                    nn.ReLU(inplace=True),
                ]
            dec_in_ch = dec_out_ch

        self.decoder = nn.Sequential(*decoder_blocks)

    # ---------------------------------------------------------------- #
    # Forward                                                           #
    # ---------------------------------------------------------------- #

    def forward(
        self,
        x_masked: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the full image from a masked input.

        The generator receives both the masked RGB and the mask so it
        knows exactly which pixels it must hallucinate.

        Args:
            x_masked: Masked image  (B, 3, H, W), hole pixels are 0.
            mask:     Binary mask   (B, 1, H, W), 1 = hole, 0 = known.

        Returns:
            G_out: Raw generator prediction (B, 3, H, W), values in [0, 1].
                   This covers the *full* image.  Call :meth:`complete` to
                   paste known pixels from ``x_masked`` back over it.
        """
        # Concatenate along channel axis so the encoder sees both colour
        # content and the hole location at every spatial position.
        inp = torch.cat([x_masked, mask], dim=1)  # (B, 4, H, W)

        h = self.encoder(inp)          # (B, feature_channels, 4, 4)
        h = h.flatten(start_dim=1)     # (B, flat_dim)

        # Deterministic bottleneck: compress → expand.
        h = self.fc_encode(h)          # (B, bottleneck_dim)
        h = self.fc_decode_up(h)       # (B, flat_dim)

        # Reshape back to a spatial feature map before the decoder.
        h = h.view(
            h.size(0),
            self.feature_channels,
            self.feature_spatial,
            self.feature_spatial,
        )                              # (B, feature_channels, 4, 4)

        G_out = self.decoder(h)        # (B, 3, H, W)
        return G_out

    # ---------------------------------------------------------------- #
    # Compositing helper                                                #
    # ---------------------------------------------------------------- #

    @staticmethod
    def complete(
        x_masked: torch.Tensor,
        mask: torch.Tensor,
        G_out: torch.Tensor,
    ) -> torch.Tensor:
        """Composite generator output with original known pixels.

        ``x_completed = x_masked + mask * G_out``

        Known pixels (mask == 0) are copied exactly from ``x_masked`` —
        the generator is never allowed to modify them.
        Hole pixels  (mask == 1) come entirely from ``G_out``.

        This is the *only* correct way to build the final completion:
        using G_out directly would silently overwrite known pixels and
        make the reconstruction loss measure the wrong thing.

        Args:
            x_masked: Masked input image   (B, 3, H, W).
            mask:     Binary mask          (B, 1, H, W), 1 = hole.
            G_out:    Generator prediction (B, 3, H, W), values in [0, 1].

        Returns:
            x_completed: Completed image   (B, 3, H, W), values in [0, 1].
        """
        return x_masked + mask * G_out


# ------------------------------------------------------------------ #
# Discriminator                                                        #
# ------------------------------------------------------------------ #

class Discriminator(nn.Module):
    """PatchGAN discriminator for image inpainting.

    Instead of classifying the whole image as real or fake, PatchGAN
    slides a convolutional classifier over overlapping patches and returns
    one logit per patch.  A positive logit means "this patch looks real";
    a negative logit means "this patch looks fake".

    Why patches?
    - Patch-level feedback is more informative than a single scalar: the
      generator gets a spatial gradient map highlighting exactly which
      regions look unrealistic.
    - A patch discriminator has far fewer parameters than a full-image one
      and is less prone to mode-collapse.

    Input:  3-channel RGB (real ground truth OR x_completed from G).
    Output: (B, 1, h, w) logit map — **no Sigmoid** — use BCEWithLogitsLoss
            which fuses sigmoid + BCE in one numerically stable operation.

    Args:
        base_channels: Channels in the first block; doubled each block.
    """

    # Four strided-conv blocks → 1-channel logit conv.
    # For a 256×256 input this gives a ~8×8 logit map, where each
    # score has a ~70×70 pixel receptive field (standard "70×70 PatchGAN").
    _NUM_BLOCKS = 4

    def __init__(self, base_channels: int) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = 3

        for i in range(self._NUM_BLOCKS):
            out_ch = base_channels * (2 ** i)  # 32, 64, 128, 256 …

            # Strided conv halves spatial dims (same k/s/p as the generator).
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1,
                          bias=False)
            )

            if i > 0:
                # InstanceNorm normalises each sample independently, so
                # the stats of real and fake images never get mixed inside
                # a batch — important when the discriminator sees both.
                # affine=True keeps learnable scale/shift per channel.
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            # First layer: no norm — absolute pixel intensities matter here.

            # LeakyReLU (slope=0.2) is standard for discriminators.
            # Unlike ReLU it lets small negative activations through,
            # which helps gradients flow even when units are "off".
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        # Final 1-channel conv → patch score map.
        # No normalisation, no activation: we output raw logits.
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=2, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute patch-level logit scores.

        Args:
            x: RGB image batch (B, 3, H, W), values in [0, 1].
               Can be real images or generator completions.

        Returns:
            logits: Patch score map (B, 1, h, w).
                    Positive → patch looks real to the discriminator.
                    Negative → patch looks fake.
                    Pass directly to BCEWithLogitsLoss; do NOT apply sigmoid
                    before the loss (it is included inside the loss for
                    numerical stability).
        """
        return self.model(x)


# ------------------------------------------------------------------ #
# Factory helpers                                                      #
# ------------------------------------------------------------------ #

def build_generator(cfg: SimpleNamespace) -> Generator:
    """Construct a Generator from a config namespace.

    Args:
        cfg: Namespace with ``image_size``, ``bottleneck_dim``,
             ``base_channels``.

    Returns:
        Initialised Generator on CPU.  Move to GPU after calling this.
    """
    return Generator(
        image_size=cfg.image_size,
        bottleneck_dim=cfg.bottleneck_dim,
        base_channels=cfg.base_channels,
    )


def build_discriminator(cfg: SimpleNamespace) -> Discriminator:
    """Construct a Discriminator from a config namespace.

    Args:
        cfg: Namespace with ``base_channels``.

    Returns:
        Initialised Discriminator on CPU.  Move to GPU after calling this.
    """
    return Discriminator(base_channels=cfg.base_channels)
