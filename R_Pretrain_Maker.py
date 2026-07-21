import numpy as np
import pandas as pd
from itertools import product
import random
import os

# =====================================================
# 1. Music Theory Constants
# =====================================================
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

CHORD_QUALITIES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "hdim7": [0, 3, 6, 10]
}

DEGREE_MAP = {
    "I": 0, "ii": 2, "iii": 4, "IV": 5,
    "V": 7, "vi": 9, "vii": 11,
    "bVII": 10, "#i": 1,
    "V/V": 2, "V/ii": 9
}

FUNCTIONAL_SUBS = {
    "I": ["vi"],
    "vi": ["I"],
    "ii": ["IV"],
    "V": ["vii"]
}

# =====================================================
# 2. Base Progressions
# =====================================================
BASE_PROGRESSIONS = [
    # Pop / Rock
    [("I", "maj"), ("V", "maj"), ("vi", "min"), ("IV", "maj"), ("I", "maj")],
    [("I", "maj"), ("vi", "min"), ("IV", "maj"), ("V", "maj"), ("I", "maj")],
    [("vi", "min"), ("IV", "maj"), ("I", "maj"), ("V", "maj"), ("I", "maj")],

    # Jazz / funcional
    [("ii", "min7"), ("V", "7"), ("I", "maj7"), ("I", "maj7"), ("I", "maj7")],
    [("I", "maj7"), ("vi", "min7"), ("ii", "min7"), ("V", "7"), ("I", "maj7")],

    # Dominantes secundarios
    [("I", "maj"), ("V/V", "7"), ("V", "7"), ("I", "maj"), ("I", "maj")],

    # Modal / cromático
    [("I", "maj"), ("bVII", "maj"), ("IV", "maj"), ("I", "maj"), ("I", "maj")]
]

# =====================================================
# 3. AUX FUNCTIONS
# =====================================================
def chord_vector(root_pc, quality):
    vec = np.zeros(12)
    for i in CHORD_QUALITIES[quality]:
        vec[(root_pc + i) % 12] = 1
    return vec

def invert_chord(vec, k):
    return np.roll(vec, k)

def apply_substitution(degree):
    if degree in FUNCTIONAL_SUBS and random.random() < 0.4:
        return random.choice(FUNCTIONAL_SUBS[degree])
    return degree

def expand_progression(prog):
    variants = []
    variants.append(prog)                      # 5
    variants.append(prog[:-1])                 # 4
    variants.append(prog + [prog[-1]])         # 6
    return variants

# =====================================================
# 4. ARMONIC COMPLEXITY
# =====================================================
def harmonic_complexity(seq):
    density = np.mean([v.sum() for v in seq])
    motion = np.mean([np.linalg.norm(seq[i] - seq[i - 1], ord=1) for i in range(1, len(seq))])
    seventh_ratio = np.mean([v.sum() >= 4 for v in seq])
    extended_notes_ratio = np.mean([v.sum() > 4 for v in seq])  # nuevas notas extra
    return density + 0.5 * motion + seventh_ratio + 0.5 * extended_notes_ratio


# =====================================================
# 5. DATA GENERATION
# =====================================================
rows = []

for base_prog in BASE_PROGRESSIONS:
    for prog in expand_progression(base_prog):
        for transpose in range(12):
            for quality_variant in [False, True]:
                for inversion in [0, 1, 2]:

                    seq_vectors = []

                    for degree, quality in prog:
                        degree = apply_substitution(degree)
                        pc = (DEGREE_MAP[degree] + transpose) % 12

                        q = quality
                        if quality_variant:
                            if quality == "maj":
                                q = "maj7"
                            elif quality == "min":
                                q = "min7"

                        vec = chord_vector(pc, q)
                        vec = invert_chord(vec, inversion)
                        seq_vectors.append(vec)

                    # Padding / cropping a 5 acordes
                    if len(seq_vectors) < 5:
                        seq_vectors += [seq_vectors[-1]] * (5 - len(seq_vectors))
                    elif len(seq_vectors) > 5:
                        seq_vectors = seq_vectors[:5]

                    comp = harmonic_complexity(seq_vectors)

                    row = {}
                    for i, v in enumerate(seq_vectors):
                        row[f"Chord_{i+1}"] = "[" + " ".join(map(str, v.astype(int))) + "]"
                    row["Complexity"] = comp
                    rows.append(row)

# =====================================================
# 6. DATAFRAME + BINNING
# =====================================================
df = pd.DataFrame(rows)

df["Bin"] = pd.qcut(
    df["Complexity"],
    q=30,
    labels=False,
    duplicates="drop"
)

df.drop(columns=["Complexity"], inplace=True)

# =====================================================
# 7. SAVE CSV
# =====================================================
dateandtime = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"PR_Chord_Sequences_w_Harmonic_Complexity_{dateandtime}.csv", index=False)
#df.to_csv("PR_Chord_Sequences_w_Harmonic_Complexity.csv", index=False)

print("Dataset redy:", df.shape)
print(df.groupby("Bin").size())
