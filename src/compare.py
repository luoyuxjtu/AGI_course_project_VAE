"""
Cross-experiment comparison and analysis.

Reads eval_metrics.json and image outputs from each experiment under
outputs/, then writes:

  outputs/comparison/beta_comparison.png
      Side-by-side generated samples and reconstructions for the three
      beta experiments (baseline β=1 / beta_0.5 β=0.5 / beta_4 β=4).
      Higher β pushes the posterior closer to the prior (better latent
      structure) at the cost of blurrier reconstructions — the grid makes
      this trade-off visually obvious.

  outputs/comparison/summary.md
      Markdown table of final val losses, parameter counts, and optional
      FID for all four experiments (baseline, beta_0.5, beta_4, lite).
      Includes a note highlighting the parameter reduction of the lite model.

Missing experiments or image files are skipped with a warning;
the script never raises an unhandled exception.

Usage
-----
    python -m src.compare
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

OUTPUTS_DIR = Path("outputs")
COMP_DIR = OUTPUTS_DIR / "comparison"

# Experiments included in the beta comparison panel
BETA_EXPS = ["baseline", "beta_0.5", "beta_4"]
BETA_LABELS = {
    "baseline": "Baseline  β=1.0",
    "beta_0.5": "β=0.5  (lower KL weight)",
    "beta_4":   "β=4.0  (higher KL weight)",
}

# All experiments used in the summary table
ALL_EXPS = ["baseline", "beta_0.5", "beta_4", "lite"]


# ------------------------------------------------------------------ #
# I/O helpers                                                          #
# ------------------------------------------------------------------ #

def _load_metrics(exp_name: str) -> Optional[dict]:
    """Load eval_metrics.json for one experiment.

    Returns the dict on success, or None with a warning on failure.
    """
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
    """Fill an axes with a grey box and a centred message."""
    ax.set_facecolor("#cccccc")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=9, color="#555555",
    )
    ax.axis("off")


# ------------------------------------------------------------------ #
# Beta comparison figure                                               #
# ------------------------------------------------------------------ #

def _make_beta_comparison() -> None:
    """Create a 2-row × 3-column comparison figure for the beta experiments.

    Row 0 — Generated samples (samples.png from each experiment)
    Row 1 — Reconstructions  (reconstructions.png from each experiment)

    Columns correspond to baseline / beta_0.5 / beta_4 in that order.
    A higher β encourages a more regular latent space (better sample
    diversity) but tends to blur reconstructions; this layout shows the
    trade-off directly.
    """
    print("\n[compare] building beta_comparison.png …")

    IMAGE_TYPES = [
        ("samples.png",         "Generated samples (prior N(0,I))"),
        ("reconstructions.png", "Reconstructions (val set, top=original / bottom=recon)"),
    ]
    n_rows = len(IMAGE_TYPES)
    n_cols = len(BETA_EXPS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for col, exp_name in enumerate(BETA_EXPS):
        for row, (img_file, row_label) in enumerate(IMAGE_TYPES):
            ax = axes[row][col]
            img_path = OUTPUTS_DIR / exp_name / img_file

            img = _load_image(img_path)
            if img is not None:
                ax.imshow(img)
                ax.axis("off")
            else:
                _placeholder_axes(
                    ax,
                    f"{exp_name}\n{img_file}\nnot available",
                )
                print(f"  [compare] missing: {img_path}")

            # Column header on the top row only
            if row == 0:
                ax.set_title(
                    BETA_LABELS.get(exp_name, exp_name),
                    fontsize=12, fontweight="bold", pad=8,
                )
            # Row label on the left column only
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=6)

    fig.suptitle(
        "β-VAE comparison: effect of KL weight on generation quality",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()

    COMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMP_DIR / "beta_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ------------------------------------------------------------------ #
# Summary markdown table                                               #
# ------------------------------------------------------------------ #

def _fmt(value, fmt=".2f", missing="—") -> str:
    """Format a numeric value, or return a placeholder for None."""
    if value is None:
        return missing
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _make_summary_md() -> None:
    """Write a markdown summary table for all experiments.

    Columns: experiment, β, base_channels, n_params, val_recon,
             val_kl, val_total, FID.

    A section below the table calls out the parameter count difference
    between the full model (base_channels=32) and the lite model
    (base_channels=16).
    """
    print("\n[compare] building summary.md …")

    rows: list[dict] = []
    for exp_name in ALL_EXPS:
        m = _load_metrics(exp_name)
        if m is not None:
            rows.append(m)

    if not rows:
        print("  [compare] No experiment metrics found; summary.md not written.")
        return

    # ---- Markdown table ------------------------------------------
    header = (
        "| Experiment | β | base_channels | n_params"
        " | val_recon | val_kl | val_total | FID |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    data_lines: list[str] = []
    for m in rows:
        fid_str = _fmt(m.get("fid"), ".2f", "—")
        line = (
            f"| {m.get('exp_name', '?')} "
            f"| {_fmt(m.get('beta'), '.1f')} "
            f"| {m.get('base_channels', '?')} "
            f"| {m.get('n_params', 0):,} "
            f"| {_fmt(m.get('final_val_recon'), '.2f')} "
            f"| {_fmt(m.get('final_val_kl'), '.2f')} "
            f"| {_fmt(m.get('final_val_total'), '.2f')} "
            f"| {fid_str} |"
        )
        data_lines.append(line)

    # ---- Parameter count note ------------------------------------
    full_params = next(
        (m.get("n_params") for m in rows if m.get("base_channels", 0) == 32),
        None,
    )
    lite_params = next(
        (m.get("n_params") for m in rows if m.get("exp_name") == "lite"),
        None,
    )
    if full_params and lite_params:
        reduction_pct = 100.0 * (full_params - lite_params) / full_params
        param_note = (
            "\n## Parameter count comparison\n\n"
            f"| Model | base_channels | n_params |\n"
            f"|---|---|---|\n"
            f"| Full (baseline / beta_0.5 / beta_4) | 32 | **{full_params:,}** |\n"
            f"| Lite | 16 | **{lite_params:,}** |\n\n"
            f"The lite model uses **{full_params - lite_params:,}** fewer parameters "
            f"(**{reduction_pct:.1f}%** reduction).\n"
        )
    else:
        param_note = "\n*(Parameter comparison unavailable — run baseline and lite experiments first.)*\n"

    # ---- Beta observation note -----------------------------------
    beta_note = (
        "\n## Effect of β on reconstruction vs latent structure\n\n"
        "- **β < 1** (beta_0.5): lower KL weight → less regularisation "
        "→ sharper reconstructions, but the latent space may be less smooth "
        "and prior samples may look less realistic.\n"
        "- **β = 1** (baseline): standard VAE balance.\n"
        "- **β > 1** (beta_4): stronger KL pressure → latent codes cluster "
        "closer to N(0,I) → more coherent prior samples, but reconstructions "
        "tend to be blurrier (information is squeezed out by the KL term).\n\n"
        "See `outputs/comparison/beta_comparison.png` for a visual comparison.\n"
    )

    # ---- Assemble and write --------------------------------------
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COMP_DIR / "summary.md"

    with open(out_path, "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(header)
        f.write("\n".join(data_lines))
        f.write("\n")
        f.write(param_note)
        f.write(beta_note)

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

    _make_beta_comparison()
    _make_summary_md()

    print(f"\n[compare] Done.  Outputs in {COMP_DIR}\n")


if __name__ == "__main__":
    main()
