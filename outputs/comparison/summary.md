# Experiment Summary

| Experiment | β | base_channels | n_params | val_recon | val_kl | val_total | FID |
|---|---|---|---|---|---|---|---|
| baseline | 1.0 | 32 | 72,706,947 | 1750.59 | 608.10 | 2358.69 | — |
| beta_0.5 | 0.5 | 32 | 72,706,947 | 1484.68 | 937.23 | 1953.30 | — |
| beta_4 | 4.0 | 32 | 72,706,947 | 2697.37 | 1037.20 | 6846.17 | — |
| lite | 1.0 | 16 | 30,767,555 | 1829.32 | 593.83 | 2423.16 | — |

## Parameter count comparison

| Model | base_channels | n_params |
|---|---|---|
| Full (baseline / beta_0.5 / beta_4) | 32 | **72,706,947** |
| Lite | 16 | **30,767,555** |

The lite model uses **41,939,392** fewer parameters (**57.7%** reduction).

## Effect of β on reconstruction vs latent structure

- **β < 1** (beta_0.5): lower KL weight → less regularisation → sharper reconstructions, but the latent space may be less smooth and prior samples may look less realistic.
- **β = 1** (baseline): standard VAE balance.
- **β > 1** (beta_4): stronger KL pressure → latent codes cluster closer to N(0,I) → more coherent prior samples, but reconstructions tend to be blurrier (information is squeezed out by the KL term).

See `outputs/comparison/beta_comparison.png` for a visual comparison.
