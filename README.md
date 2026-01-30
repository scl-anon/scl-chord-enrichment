
# SCL: Semi-Supervised CVAE-LSTM for Chord Progression Enrichment

This repository contains the official implementation of the paper:

**“SCL: Semi-Supervised CVAE-LSTM for Chord Progression Enrichment Conditioned on Harmonic Complexity”**

---

## Abstract
Chord progression reharmonization is a common practice in various musical contexts, including reinterpretations, live arrangements,
and creative performances. The goal is to enrich simple harmonic sequences with more complex and harmonically richer variations. In
this work, we propose SCL, a semi-supervised Conditional Variational Autoencoder (CVAE) with LSTM layers, trained on symbolic music
data in MIDI format. SCL transforms basic progressions into harmonically enriched versions by applying musical transformations—such
as added tensions and functional substitutions—while maintaining tonal coherence. Unlike existing approaches, SCL does not depend
on the initial melody but uses only the input chord progression, enabling a structured and interpretable representation of chords and
their modifications. We evaluate SCL on a dataset of manually enriched MIDI progressions, using data augmentation via transposition.
Results show that SCL generates harmonically rich and perceptually complex progressions, achieving performance comparable to
human reharmonizations and surpassing some approaches reported in the literature, under both objective and subjective evaluations.
---

## Repository structure

```text
.
├── environment.yml                  # Conda environment for reproducibility
├── GeneratorSCL.py                  # Main generation script (SCL model)
├── TrainSCL.ipynb                   # Training notebook (SCL model)
├── RVAE.ipynb                       # RVAE baseline notebook
│     
├── Models/                          # Trained models and vocabularies
│   ├── scl_fulltrained.pt           # Checkpoint used in the paper
│   ├── token_to_id_train.pkl
│   └── id_to_token_train.pkl
│
├── MIDI_TEST/                       # Generated MIDI examples (paper results)
│   ├── Simple_Progressions/         # Input progressions
│   ├── Human_Reharmonization/       # Human reharmonizations
│   ├── RVAE/                        # RVAE outputs
│   └── SCL/                         # SCL outputs (different configs)
│
├── Listening_Test/                  # Subjective evaluation
│   ├── Listening_Test.pdf           # Instructions and questionnaire
│   ├── Listening_Test_results.csv   # Collected responses
│   ├── gms_subscales_scoring.xls    # GMS guied
│   └── WAVs/                        # Audio used in the test
│
├── metrics.csv                      # Objective metrics (Table 6)
├── metrics_sint.csv                 # Synthetic metrics (Table 8)
│
├── PRE_TRAIN.zip                    # Pretraining dataset (X → X)
├── MIDI_TRAIN.zip                   # Supervised training dataset (X → Y)
├── R_Pretrain.csv                   # R_Pretrain dataset
├── R_Pretrain_Maker.py              # Script to create R_Pretrain.csv
│
└── README.md
```

The datasets provided as ZIP files are already preprocessed
and correspond exactly to those used in the experiments.

---
## Requirements

- Conda (Miniconda or Anaconda)
- Python 3.10
- PyTorch
- numpy
- pretty_midi
- music21
- pandas
- scikit-learn
---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate scl
```

The environment is intentionally minimal and installs only the dependencies required for inference, ensuring fast setup and reproducibility.

---

## How to Run (SCL Generator)

The main generation script is **`GeneratorSCL.py`**.  
It processes a **folder of MIDI files** containing simple chord progressions and generates harmonically enriched versions using the trained SCL model.

### Command-line usage

```bash
python GeneratorSCL.py <input_folder> <seed> <sigma> <N>
```

Arguments:
- `<input_folder>`: Path to the folder containing input MIDI files with simple chord progressions
- `<seed>`: Random seed for reproducibility (integer)
- `<sigma>`: Standard deviation for sampling in the latent space (float)
- `<N>`: Number of enriched variations to generate per input file (integer)

### Example

```bash
python GeneratorSCL.py MIDI_TEST/Simple_Progressions/ 42 0.5 3
``` 

This command generates enriched chord progressions for all MIDI files in the specified folder using the trained SCL model.

---

### Output
Generated MIDI files are written to disk under:

```
./generated_midis/
└── <input_name>_seed<seed>_sigma<sigma>_N<N>_<timestamp>/
    ├── 1_SCL_...
    ├── 2_SCL_...
    └── ...
```

Each output MIDI corresponds to the best candidate selected among N latent samples for a given input progression.

---

## Generation and Selection Strategy

For each input progression:

1. The progression is encoded using the trained CVAE-LSTM encoder.
2. N latent samples are drawn as:
    $$z = μ + σ * ε$$
    where $ε ~ N(0, I)$
3. Each latent sample is decoded into a candidate chord sequence.
4. Candidates identical to the input are discarded.
5. The remaining candidates are ranked using a fast harmonic reward.
6. The candidate with the lowest reward value is selected and saved.

---

