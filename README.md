# SCL: Semi-supervised VAE-LSTM for Chord Progression Enrichment

Official repository for the paper submitted to ICTAI 2026.

## Structure

```
paper/                    # LaTeX source (IEEE format) + bibliography
main.pdf                  # Compiled paper
data/
  midi/
    simple/               # 6 simple input progressions
    scl_default/          # 6 SCL outputs (default: tau=1.0, N=5)
    scl_tuned/            # 6 SCL outputs (tuned: tau=1.26, N=19)
    human/                # 6 human reharmonizations
    rvae/                 # 6 RVAE outputs
    baselines/random/     # 36 random baseline MIDIs (6 seeds)
  survey/                 # Anonymous survey responses (GMSI + PHC)
  features/               # jSymbolic features, relatedness metrics
  csv/                    # Training datasets (R_Pretrain, metrics)
src/
  model/                  # CVAE-LSTM model architecture + generation
  checkpoints/            # Trained weights + chord vocabulary
  evaluation/             # Relatedness, baselines, survey, case study
experiments/
  logs/                   # Training loss logs (Phase 1 + Phase 2)
  case_study/             # MIDIs for qualitative analysis
  listening_test/         # WAV files used in subjective evaluation
training/                 # Training notebook + pre-trained model + data
jsymbolic/                # jSymbolic 2.2 JAR for feature extraction
```

## Key Results (Table 1)

| Condition | VDR (complexity) | Relatedness |
|-----------|:---:|:---:|
| Simple    | 0.013 | 1.000 |
| Human     | 0.314 | 0.683 |
| RVAE      | 0.178 | 0.377 |
| SCL Default | 0.292 | 0.566 |
| SCL Tuned   | 0.270 | 0.513 |
| Random    | 0.383 | ~0.0  |

SCL achieves higher input--output relatedness than RVAE (0.566 vs 0.377) while maintaining comparable harmonic complexity. The Random baseline scores highest on most complexity metrics, confirming that complexity alone is insufficient as an evaluation criterion.

## Reproducing Results

```bash
# Relatedness metrics (Table 1, Relatedness row)
python3 src/evaluation/relatedness.py

# Random baseline features with mean ± std
python3 src/evaluation/random_baseline_stats.py

# Feature extraction for baselines
python3 src/evaluation/baseline_features.py

# Survey re-analysis
python3 src/evaluation/survey_reanalysis.py

# Case study
python3 src/evaluation/case_study.py
```

## Model

- Architecture: VAE-LSTM (single LSTM layer, hidden=64, latent=64)
- Training: two-phase (600 epochs pretrain + 100 epochs supervised)
- Vocabulary: 310 chord tokens
- Checkpoint: `src/checkpoints/cvae_lstm_model.pth`

## License

This repository is provided for research purposes. Contact the authors for usage permissions.
