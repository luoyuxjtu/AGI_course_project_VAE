"""
ELBO loss for the VAE.

The Evidence Lower BOund (ELBO) is:
    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))

We maximise the ELBO, which is equivalent to minimising:
    loss = recon + beta * KL

where:
  recon  — reconstruction term  (negative log-likelihood proxy)
  KL     — KL divergence between the approximate posterior q(z|x)
            and the prior p(z) = N(0, I)
  beta   — weighting factor (beta=1 → standard VAE; beta>1 → beta-VAE)
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def elbo_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE ELBO loss and its two components.

    Reconstruction term — MSE, summed over pixels, averaged over batch
    -----------------------------------------------------------------------
    Using MSE here is equivalent to assuming a Gaussian observation model
        p(x | z) = N(x;  x_recon,  sigma^2 * I)
    with a fixed (non-learned) variance sigma^2.  Under that model,
    -log p(x|z) = (1 / 2*sigma^2) * ||x - x_recon||^2 + const,
    so minimising the pixel-sum MSE maximises this Gaussian log-likelihood.
    This Gaussian assumption is one of the theoretical points discussed
    in Module 3 (Theoretical Analysis).

    KL term — closed-form for diagonal Gaussians
    -----------------------------------------------------------------------
    For q(z|x) = N(mu, diag(exp(logvar))) and p(z) = N(0, I):
        KL = -0.5 * sum_j(1 + logvar_j - mu_j^2 - exp(logvar_j))
    This is summed over the latent dimensions and averaged over the batch.
    KL >= 0 always; equality holds when q = p (mu=0, logvar=0).

    Args:
        x_recon: Reconstructed images from the decoder  (B, C, H, W), [0,1].
        x:       Original input images                  (B, C, H, W), [0,1].
        mu:      Posterior mean from the encoder         (B, latent_dim).
        logvar:  Posterior log-variance from the encoder (B, latent_dim).
        beta:    KL weight.  beta=1 → standard VAE; beta>1 → beta-VAE
                 (encourages a more disentangled / regularised latent space).

    Returns:
        recon: Reconstruction loss scalar  (pixel-sum MSE, batch-averaged).
        kl:    KL divergence scalar        (latent-sum KL,  batch-averaged).
        total: Combined loss scalar        (recon + beta * kl).
    """
    batch_size = x.size(0)

    # --- Reconstruction loss ------------------------------------------
    # reduction='sum' accumulates over all elements (B*C*H*W); dividing by
    # batch_size gives the per-sample pixel-sum MSE.
    recon = F.mse_loss(x_recon, x, reduction="sum") / batch_size

    # --- KL divergence ------------------------------------------------
    # Analytical KL between N(mu, exp(logvar)) and N(0, I).
    # Sum over latent dimensions (dim=1), then average over the batch.
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

    # --- Total ELBO loss ----------------------------------------------
    total = recon + beta * kl

    return recon, kl, total
