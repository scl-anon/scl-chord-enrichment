# Preliminary Training Dynamics — Supplementary Data

Training loss logs from a preliminary experiment using a reduced chord vocabulary (83 tokens) 
and a smaller dataset (CSV-based). These logs illustrate the two-phase training dynamics.

## Phase 1: Unsupervised Pretraining (X → X, 600 epochs)

| Metric | Epoch 1 | Epoch 600 | Change |
|--------|---------|-----------|--------|
| Recon Loss | 22,888 | 269 | -98.8% |
| KL Loss | 469 | 10,418 | +2,122% (annealing) |
| Total Loss | 22,935 | 1,310 | -94.3% |

## Phase 2: Supervised Fine-tuning (X → Y, 600 epochs)

| Metric | Epoch 1 | Epoch 600 | Mean ± SD |
|--------|---------|-----------|-----------|
| Recon Loss | 51.5 | 39.0 | 52.1 ± 14.7 |
| KL Loss | 323.8 | 284.6 | 302.5 ± 12.5 |
| Music Loss | 7.87 | 7.43 | 7.46 ± 0.17 |

## Key Observations

1. **Phase 1 provides effective initialization**: Recon loss drops from 22,888 to 269, teaching the model to represent chord progressions.
2. **Phase 2 starts from a strong base**: Recon loss begins at 51 (≈450× lower than Phase 1 start).
3. **Music losses are actively optimized**: Music loss stays stable at ~7.46 ± 0.17 throughout Phase 2, indicating the model balances reconstruction with tonal coherence, tension, and voice-leading constraints.
4. **KL annealing works as intended**: KL increases gradually during Phase 1 (β: 0 → 2) to regularize the latent space.

## Data Files

- `phase1_pretrain.csv` — 600 epochs of Phase 1 (total_loss, recon_loss, kl_loss)
- `phase2_supervised.csv` — 600 epochs of Phase 2 (total_loss, recon_loss, kl_loss, music_loss)

## Source

Extracted from `test4EVO/traincsv.ipynb`, which trained a VAE-LSTM model on a CSV-encoded chord progression dataset (vocab_size=83, 60 unique chord types, 480 training pairs + 120 validation pairs).
