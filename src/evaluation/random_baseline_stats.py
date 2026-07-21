"""
Compute Random baseline features with mean ± std for Table 1.
Generates 36 random MIDI files (6 seeds × 6 test inputs) and extracts 
the 11 jSymbolic-equivalent features used in the paper.

Run: python3 src/evaluation/random_baseline_stats.py
"""
import os
import sys
import pickle
import random
import numpy as np
import pretty_midi

FEATURE_NAMES = ["VMS", "VT", "VS", "VDR", "ST", "7C", "NSC", "CC",
                 "DTMCVI", "PRTMCVI", "VNSPC"]
N_SEEDS = 6
SEEDS = [100, 200, 300, 400, 500, 600]


def load_vocabulary():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(base, "src", "checkpoints", "token_to_id_train.pkl"), "rb") as f:
        token_to_id = pickle.load(f)
    return [k for k in token_to_id if k not in ("<pad>", "<sos>", "<eos>")]


def count_chords_in_midi(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda x: x.start)
    if not notes:
        return 0
    n, t0 = 1, notes[0].start
    for note in notes[1:]:
        if note.start - t0 >= 0.05:
            n += 1
            t0 = note.start
    return n


def token_to_midi_notes(token):
    try:
        return [60 + int(x) for x in token.split("_")]
    except (ValueError, AttributeError):
        return []


def create_random_midi(tokens, output_path):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    t = 0.0
    for tok in tokens:
        for pitch in token_to_midi_notes(tok):
            piano.notes.append(pretty_midi.Note(
                velocity=80, pitch=pitch, start=t, end=t + 2.0))
        t += 2.0
    midi.instruments.append(piano)
    midi.write(output_path)


def extract_features(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda x: x.start)
    if not notes:
        return {k: 0.0 for k in FEATURE_NAMES}

    chords, current, t0 = [], set(), None
    for note in notes:
        if t0 is None:
            t0 = note.start
            current = {note.pitch % 12}
        elif note.start - t0 < 0.05:
            current.add(note.pitch % 12)
        else:
            if current:
                chords.append(tuple(sorted(current)))
            t0 = note.start
            current = {note.pitch % 12}
    if current:
        chords.append(tuple(sorted(current)))
    if not chords:
        return {k: 0.0 for k in FEATURE_NAMES}

    n_chords = len(chords)
    total_intervals = 0
    interval_counts = {}

    for ch in chords:
        pcs = sorted(ch)
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                ic = (pcs[j] - pcs[i]) % 12
                interval_counts[ic] = interval_counts.get(ic, 0) + 1
                total_intervals += 1

    ifr = lambda ic: interval_counts.get(ic, 0) / max(total_intervals, 1)

    result = {
        "VMS": ifr(1),
        "VT": ifr(6),
        "VS": ifr(10) + ifr(11),
        "VDR": sum(ifr(ic) for ic in [1, 2, 6, 10, 11]),
        "ST": sum(1 for c in chords if len(c) == 3) / n_chords,
        "7C": sum(1 for c in chords if len(c) == 4) / n_chords,
        "NSC": sum(1 for c in chords if len(c) not in (3, 4)) / n_chords,
        "CC": sum(1 for c in chords if len(c) >= 5) / n_chords,
    }

    top2 = sorted(interval_counts.items(), key=lambda x: -x[1])[:2]
    result["DTMCVI"] = abs(top2[0][0] - top2[1][0]) if len(top2) >= 2 else 0.0
    result["PRTMCVI"] = (top2[0][1] / max(top2[1][1], 1)) if len(top2) >= 2 else 1.0
    result["VNSPC"] = float(np.std([len(c) for c in chords])) if n_chords > 1 else 0.0

    return result


def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(base, "data", "midi", "input")
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mid")])

    chord_tokens = load_vocabulary()
    print(f"Vocabulary: {len(chord_tokens)} chord types")
    print(f"Test inputs: {len(input_files)}")
    print(f"Seeds: {N_SEEDS}")
    print(f"Total random MIDIs to generate: {N_SEEDS * len(input_files)}\n")

    # Collect features across all seeds and inputs
    all_features = {fn: [] for fn in FEATURE_NAMES}

    for seed in SEEDS:
        random.seed(seed)
        for fname in input_files:
            n_chords = count_chords_in_midi(os.path.join(input_dir, fname))
            tokens = random.choices(chord_tokens, k=n_chords)
            tmp_path = os.path.join(base, "data", "baselines", "random",
                                    f"RANDOM_SEED{seed}_{fname}")
            create_random_midi(tokens, tmp_path)
            feats = extract_features(tmp_path)
            for k in FEATURE_NAMES:
                all_features[k].append(feats[k])

    # Print results in LaTeX table format
    print("=" * 70)
    print("RANDOM BASELINE — Mean ± Std (6 seeds × 6 inputs = 36 MIDIs)")
    print("=" * 70)
    for k in FEATURE_NAMES:
        vals = all_features[k]
        m, s = np.mean(vals), np.std(vals)
        arrow = r"$\uparrow$" if k != "ST" and k != "PRTMCVI" else r"$\downarrow$"
        print(f"{k} {arrow} & {m:.3f} $\\pm$ {s:.3f} \\\\")

    print("\n--- LaTeX row for Table 1 ---")
    for k in FEATURE_NAMES:
        vals = all_features[k]
        m, s = np.mean(vals), np.std(vals)
        print(f"{k}: & {m:.3f} $\\\\pm$ {s:.3f}")

    print("\n--- Summary CSV ---")
    csv_path = os.path.join(base, "data", "baselines", "random_stats.csv")
    with open(csv_path, "w") as f:
        f.write("feature,mean,std\n")
        for k in FEATURE_NAMES:
            vals = all_features[k]
            f.write(f"{k},{np.mean(vals):.4f},{np.std(vals):.4f}\n")
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
