"""
Cross-experiment comparison for the GAN inpainting project.

Reads outputs/{exp}/inpainting.png and eval_metrics.json from each
completed experiment and writes:

  outputs/comparison/gan_vs_recon.png
      Left column:  baseline  (full GAN, L1 + adversarial loss).
      Right column: recon_only (L1 only, no discriminator).
      Both columns show the same validation images with the same masks
      (evaluate.py uses a fixed seed), so any visible difference is
      due purely to the adversarial term — sharper vs blurrier fills.

  outputs/comparison/summary.md
      Table of val L1, PSNR, SSIM, G and D parameter counts, and
      optional FID for all three experiments (baseline, recon_only, lite).
      A section highlights the parameter reduction of the lite model.

Missing experiment outputs are skipped with a warning; the script never
raises an unhandled exception.

Usage
-----
    python -m src.compare
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

OUTPUTS_DIR = Path("outputs")
COMP_DIR    = OUTPUTS_DIR / "comparison"

# Experiments shown in the adversarial-ablation figure
GAN_VS_RECON_EXPS = ["baseline", "recon_only"]
GAN_VS_RECON_LABELS = {
    "baseline":   "Baseline GAN  (L1 + adversarial)",
    "recon_only": "Recon only  (L1 only, no discriminator)",
}

# All experiments included in the summary table
ALL_EXPS = ["baseline", "recon_only", "lite"]


# ------------------------------------------------------------------ #
# I/O helpers                                                          #
# ------------------------------------------------------------------ #

def _load_metrics(exp_name: str) -> Optional[dict]:
    """Load eval_metrics.json for one experiment, or return None."""
    path = OUTPUTS_DIR / exp_name / "eval_metrics.json"
    if not path.exists():
        print(f"  [compare] SKIP  {exp_name}: eval_metrics.json not found ({path})")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        print(f"  [compare] SKIP  {exp_name}: could not read metrics ({exc!r})")
        return None


def _load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image file as an RGB numpy array, or return None."""
    if not path.exists():
        return None
    try:
        return mpimg.imread(str(path))
    except Exception as exc:
        print(f"  [compare] could not load {path} ({exc!r})")
        return None


def _placeholder_axes(ax: plt.Axes, message: str) -> None:
    """Fill an axes with a grey box and centred message."""
    ax.set_facecolor("#cccccc")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=9, color="#555555",
    )
    ax.axis("off")


# ------------------------------------------------------------------ #
# GAN vs recon comparison figure                                       #
# ------------------------------------------------------------------ #

def _make_gan_vs_recon() -> None:
    """Two-column figure: baseline (full GAN) vs recon_only (L1 only).

    Each column is that experiment's inpainting.png, which already
    contains rows of [ masked input | completion | ground truth ].
    Because evaluate.py uses a fixed seed for mask generation and the
    val DataLoader is deterministic (shuffle=False), both columns show
    the same input images and the same hole patterns — making the
    sharpness / realism difference directly attributable to the
    adversarial loss.
    """
    print("\n[compare] building gan_vs_recon.png …")

    fig, axes = plt.subplots(
        1, len(GAN_VS_RECON_EXPS),
        figsize=(13 * len(GAN_VS_RECON_EXPS), 14),
        squeeze=False,
    )

    for col, exp_name in enumerate(GAN_VS_RECON_EXPS):
        ax = axes[0][col]
        img_path = OUTPUTS_DIR / exp_name / "inpainting.png"
        img = _load_image(img_path)

        if img is not None:
            ax.imshow(img)
            ax.axis("off")
        else:
            _placeholder_axes(ax, f"{exp_name}\ninpainting.png\nnot available")
            print(f"  [compare] missing: {img_path}")

        ax.set_title(
            GAN_VS_RECON_LABELS.get(exp_name, exp_name),
            fontsize=13, fontweight="bold", pad=10,
        )

    fig.suptitle(
        "Effect of the adversarial loss on inpainting quality\n"
        "Each panel shows:  [ masked input  |  completion  |  ground truth ]",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    COMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMP_DIR / "gan_vs_recon.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ------------------------------------------------------------------ #
# Summary markdown table                                               #
# ------------------------------------------------------------------ #

def _fmt(value, fmt: str = ".4f", missing: str = "—") -> str:
    """Format a numeric value, or return a placeholder for None / missing."""
    if value is None:
        return missing
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _make_summary_md() -> None:
    """Write summary.md with per-experiment metrics and parameter analysis."""
    print("\n[compare] building summary.md …")

    rows: list[dict] = []
    for exp_name in ALL_EXPS:
        m = _load_metrics(exp_name)
        if m is not None:
            rows.append(m)

    if not rows:
        print("  [compare] No experiment metrics found; summary.md not written.")
        return

    # ---- Main metrics table --------------------------------------
    header = (
        "| Experiment | base_ch | n_params_G | n_params_D"
        " | val L1 | PSNR (dB) | SSIM | FID |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    data_lines: list[str] = []
    for m in rows:
        n_g = m.get("n_params_g", 0)
        n_d = m.get("n_params_d", 0)
        line = (
            f"| {m.get('exp_name', '?')} "
            f"| {m.get('base_channels', '?')} "
            f"| {n_g:,} "
            f"| {n_d:,} "
            f"| {_fmt(m.get('val_l1'),  '.4f')} "
            f"| {_fmt(m.get('psnr_db'), '.2f')} "
            f"| {_fmt(m.get('ssim'),    '.4f')} "
            f"| {_fmt(m.get('fid'),     '.1f')} |"
        )
        data_lines.append(line)

    # ---- Full vs lite parameter comparison -----------------------
    # Use the first full-size experiment available (baseline preferred).
    full_m = next(
        (m for m in rows if m.get("exp_name") in ("baseline", "recon_only")),
        None,
    )
    lite_m = next((m for m in rows if m.get("exp_name") == "lite"), None)

    if full_m and lite_m:
        full_g = full_m.get("n_params_g", 0)
        full_d = full_m.get("n_params_d", 0)
        lite_g = lite_m.get("n_params_g", 0)
        lite_d = lite_m.get("n_params_d", 0)
        full_total = full_g + full_d
        lite_total = lite_g + lite_d
        pct = 100.0 * (full_total - lite_total) / full_total if full_total else 0.0
        param_note = (
            "\n## Parameter count: full vs lite\n\n"
            f"| Model | base_channels | n_params_G | n_params_D | Total |\n"
            f"|---|---|---|---|---|\n"
            f"| Full ({full_m.get('exp_name')}) "
            f"| {full_m.get('base_channels', '?')} "
            f"| **{full_g:,}** | **{full_d:,}** | **{full_total:,}** |\n"
            f"| Lite "
            f"| {lite_m.get('base_channels', '?')} "
            f"| **{lite_g:,}** | **{lite_d:,}** | **{lite_total:,}** |\n\n"
            f"Lite reduces total parameters by **{full_total - lite_total:,}**"
            f" (**{pct:.1f}%** reduction).\n"
        )
    else:
        param_note = (
            "\n*(Parameter comparison unavailable — "
            "run baseline and lite first.)*\n"
        )

    # ---- Adversarial ablation discussion -------------------------
    adv_note = (
        "\n## Effect of the adversarial loss (baseline vs recon_only)\n\n"
        "- **recon_only** (`lambda_adv = 0`): generator trained with L1 only.\n"
        "  L1 minimises the expected pixel-wise error, which averages over all\n"
        "  plausible completions and produces blurry, low-frequency fills with\n"
        "  no sharp edges or texture.\n"
        "- **baseline** (`lambda_adv = 1`): full GAN.  The discriminator penalises\n"
        "  completions that do not lie on the manifold of real images, pushing\n"
        "  the generator toward sharper textures and more realistic fills —\n"
        "  at the potential cost of slightly higher L1 (the discriminator\n"
        "  rewards realism over pixel accuracy).\n\n"
        "See `outputs/comparison/gan_vs_recon.png` for a visual comparison.\n"
    )

    # ---- Write ---------------------------------------------------
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMP_DIR / "summary.md"
    with open(out_path, "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(header)
        f.write("\n".join(data_lines))
        f.write("\n")
        f.write(param_note)
        f.write(adv_note)

    print(f"  → {out_path}")
    print(f"  Experiments summarised: {[m.get('exp_name') for m in rows]}")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    """Generate all comparison outputs."""
    print(f"\n{'='*60}")
    print("[compare] Comparing experiments under outputs/")
    print(f"{'='*60}")

    _make_gan_vs_recon()
    _make_summary_md()

    print(f"\n[compare] Done.  Outputs in {COMP_DIR}\n")


if __name__ == "__main__":
    main()
