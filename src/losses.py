"""
Loss functions for the GAN inpainting project.

Three building blocks:

reconstruction_loss   — L1 between the completed image and the ground truth.
discriminator_loss    — standard GAN D loss: real→1, fake→0.
generator_adv_loss    — non-saturating GAN G loss: fake→1.

Why L1 + adversarial?
---------------------
L1 alone minimises the expected absolute pixel error.  Because natural
images are multi-modal (many plausible textures can fill a hole), the
optimal L1 prediction is the *average* of all plausible completions —
which is a blurry, low-frequency image with no sharp edges or texture.

The adversarial loss fixes this: the discriminator penalises any
completion that does not lie on the manifold of real images, forcing the
generator to produce sharp, locally coherent predictions rather than
averaged-out grey smears.  The interplay between these two terms is the
central point of the theoretical analysis module:

  total_G = lambda_rec * L1  +  lambda_adv * adv_G

With lambda_rec >> lambda_adv (e.g. 100 vs 1) the generator stays
anchored to the correct colours/structure (L1) while the adversarial
term sharpens fine detail.  Setting lambda_adv=0 (recon_only experiment)
lets us see exactly what L1 alone produces.

Implementation note
-------------------
All adversarial losses use BCEWithLogitsLoss, which fuses the sigmoid
with the cross-entropy in a single numerically stable operation.
Never apply sigmoid to logits before passing them to this loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# One shared instance is enough — BCEWithLogitsLoss has no learnable
# parameters and is stateless, so it is safe to reuse across calls.
_bce_loss = nn.BCEWithLogitsLoss()


# ------------------------------------------------------------------ #
# Reconstruction loss                                                  #
# ------------------------------------------------------------------ #

def reconstruction_loss(
    x_completed: torch.Tensor,
    x_real: torch.Tensor,
) -> torch.Tensor:
    """L1 pixel loss between the completed image and the ground truth.

    Because known pixels are copied through exactly
    (x_completed = x_masked + mask * G_out), the L1 error is non-zero
    only inside the hole — the loss naturally concentrates on the region
    the generator had to invent.

    Args:
        x_completed: Generator completion (B, 3, H, W), values in [0, 1].
        x_real:      Ground-truth image   (B, 3, H, W), values in [0, 1].

    Returns:
        Scalar L1 loss averaged over batch, channels, and pixels.
    """
    return F.l1_loss(x_completed, x_real, reduction="mean")


# ------------------------------------------------------------------ #
# Adversarial losses                                                   #
# ------------------------------------------------------------------ #

def discriminator_loss(
    d_real_logits: torch.Tensor,
    d_fake_logits: torch.Tensor,
) -> torch.Tensor:
    """GAN discriminator loss: real patches → 1, fake patches → 0.

    D wants to tell real images apart from generator completions.
    We use the standard non-saturating formulation:

        L_D = BCE(D(x_real), 1)  +  BCE(D(x_completed.detach()), 0)

    The caller must pass ``d_fake_logits`` computed on a *detached*
    completion (``x_completed.detach()``) so that gradients from this
    loss do not flow back into the generator.

    Args:
        d_real_logits: Discriminator output on real images   (B, 1, h, w).
        d_fake_logits: Discriminator output on completions   (B, 1, h, w),
                       computed with x_completed.detach().

    Returns:
        Scalar discriminator loss (average of real and fake terms).
    """
    real_target = torch.ones_like(d_real_logits)
    fake_target = torch.zeros_like(d_fake_logits)

    loss_real = _bce_loss(d_real_logits, real_target)
    loss_fake = _bce_loss(d_fake_logits, fake_target)

    return loss_real + loss_fake


def generator_adv_loss(d_fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating GAN generator adversarial loss: fake patches → 1.

    G wants the discriminator to label its completions as real.  In the
    original GAN formulation G would minimise log(1 - D(G(x))), but this
    saturates early in training (when D is confident).  The non-saturating
    alternative minimises -log(D(G(x))), i.e. maximises the discriminator
    score for fake samples — equivalent to BCE(fake_logits, 1).

    Args:
        d_fake_logits: Discriminator output on completions (B, 1, h, w),
                       computed on x_completed *without* detach so
                       gradients flow back into the generator.

    Returns:
        Scalar adversarial loss for the generator.
    """
    real_target = torch.ones_like(d_fake_logits)
    return _bce_loss(d_fake_logits, real_target)


def generator_total_loss(
    g_recon: torch.Tensor,
    g_adv: torch.Tensor,
    lambda_rec: float,
    lambda_adv: float,
) -> torch.Tensor:
    """Combine reconstruction and adversarial losses for the generator.

    total = lambda_rec * L1  +  lambda_adv * adv

    With lambda_rec=100 and lambda_adv=1 (default), the L1 term keeps
    the completion structurally correct while the adversarial term
    sharpens texture and edges.  Setting lambda_adv=0 reduces to pure L1
    (the recon_only ablation experiment).

    Args:
        g_recon:    Reconstruction (L1) loss scalar.
        g_adv:      Adversarial loss scalar.
        lambda_rec: Weight on the reconstruction term.
        lambda_adv: Weight on the adversarial term.

    Returns:
        Combined scalar loss for the generator update step.
    """
    return lambda_rec * g_recon + lambda_adv * g_adv
