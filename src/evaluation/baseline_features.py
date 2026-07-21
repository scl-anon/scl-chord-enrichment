"""
Quick jSymbolic-equivalent feature extraction for baseline MIDI files.
Computes the 11 features used in the paper using Python (no Java needed).
"""
import os
import sys
import csv
import numpy as np
import pretty_midi
from collections import Counter


def midi_to_chord_sequence(midi_path, threshold=0.05):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda x: x.start)
    if not notes:
        return []
    chords = []
    current_chord = set()
    current_time = None
    for note in notes:
        if current_time is None:
            current_time = note.start
            current_chord = {note.pitch % 12}
        elif note.start - current_time < threshold:
            current_chord.add(note.pitch % 12)
        else:
            if current_chord:
                chords.append(tuple(sorted(current_chord)))
            current_time = note.start
            current_chord = {note.pitch % 12}
    if current_chord:
        chords.append(tuple(sorted(current_chord)))
    return chords


def compute_vertical_intervals(chords):
    """Compute all vertical interval counts across chords."""
    interval_counts = Counter()
    total_intervals = 0
    for ch in chords:
        pcs = sorted(ch)
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                interval = (pcs[j] - pcs[i]) % 12
                interval_counts[interval] += 1
                total_intervals += 1
    return interval_counts, total_intervals


def extract_features(midi_path):
    """Extract the 11 jSymbolic features used in the paper."""
    chords = midi_to_chord_sequence(midi_path)
    n_chords = len(chords)
    if n_chords == 0:
        return {k: 0.0 for k in FEATURE_NAMES}

    interval_counts, total_intervals = compute_vertical_intervals(chords)

    # Helper: get fraction for a given interval class
    def interval_fraction(ic):
        return interval_counts.get(ic, 0) / max(total_intervals, 1)

    # VMS: fraction of minor second interval
    VMS = interval_fraction(1)

    # VT: fraction of tritone interval
    VT = interval_fraction(6)

    # VS: fraction of major/minor seventh (interval 10 or 11)
    VS = interval_fraction(10) + interval_fraction(11)

    # VDR: fraction of dissonant intervals (1, 2, 6, 10, 11)
    VDR = sum(interval_fraction(ic) for ic in [1, 2, 6, 10, 11])

    # ST: fraction of chords that are triads (exactly 3 pitch classes)
    n_triads = sum(1 for ch in chords if len(ch) == 3)
    ST = n_triads / n_chords

    # 7C: fraction of chords that are seventh chords (exactly 4 pitch classes)
    n_sevenths = sum(1 for ch in chords if len(ch) == 4)
    SC = n_sevenths / n_chords  # renamed to avoid number-starting var

    # NSC: fraction of non-standard chords (not 3 or 4 notes)
    n_nonstd = sum(1 for ch in chords if len(ch) not in (3, 4))
    NSC = n_nonstd / n_chords

    # CC: fraction of complex chords (5+ notes)
    n_complex = sum(1 for ch in chords if len(ch) >= 5)
    CC = n_complex / n_chords

    # DTMCVI: Distance Between Two Most Common Vertical Intervals
    common = interval_counts.most_common(2)
    if len(common) >= 2:
        DTMCVI = abs(common[0][0] - common[1][0])
    else:
        DTMCVI = 0.0

    # PRTMCVI: Prevalence Ratio of Two Most Common Vertical Intervals
    if len(common) >= 2:
        ratio = common[0][1] / max(common[1][1], 1)
        PRTMCVI = ratio
    else:
        PRTMCVI = 1.0

    # VNSPC: Variability of Number of Simultaneous Pitch Classes (std of chord sizes)
    chord_sizes = [len(ch) for ch in chords]
    VNSPC = float(np.std(chord_sizes)) if len(chord_sizes) > 1 else 0.0

    return {
        "VMS": VMS,
        "VT": VT,
        "VS": VS,
        "VDR": VDR,
        "ST": ST,
        "7C": SC,
        "NSC": NSC,
        "CC": CC,
        "DTMCVI": DTMCVI,
        "PRTMCVI": PRTMCVI,
        "VNSPC": VNSPC,
    }


FEATURE_NAMES = ["VMS", "VT", "VS", "VDR", "ST", "7C", "NSC", "CC", "DTMCVI", "PRTMCVI", "VNSPC"]


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Directories to extract features from
    dirs = {
        "Simple": os.path.join(base_dir, "data", "midi", "input"),
        "SCL": os.path.join(base_dir, "data", "midi", "scl_output"),
        "Human": os.path.join(base_dir, "data", "midi", "human_output"),
        "Random": os.path.join(base_dir, "data", "baselines", "random"),
        "Mismatched": os.path.join(base_dir, "data", "baselines", "mismatched"),
    }

    all_results = []

    for label, directory in dirs.items():
        if not os.path.exists(directory):
            print(f"  WARNING: {directory} does not exist, skipping {label}")
            continue

        files = sorted([f for f in os.listdir(directory) if f.endswith(".mid")])
        print(f"\n=== {label} ({len(files)} files) ===")

        for f in files:
            path = os.path.join(directory, f)
            features = extract_features(path)
            features["source"] = label
            features["file"] = f
            all_results.append(features)

    # Print summary table
    print("\n=== Summary: Mean Features by Source ===")
    print(f"{'Source':<12}", end="")
    for fn in FEATURE_NAMES:
        print(f" {fn:>10}", end="")
    print()

    for label in dirs.keys():
        group = [r for r in all_results if r["source"] == label]
        if not group:
            continue
        print(f"{label:<12}", end="")
        for fn in FEATURE_NAMES:
            values = [r[fn] for r in group]
            mean = np.mean(values)
            std = np.std(values)
            print(f" {mean:>6.3f}", end="")  # simplified, no std
        print()

    # Save to CSV
    output_csv = os.path.join(base_dir, "data", "baselines", "all_features_comparison.csv")
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["source", "file"] + FEATURE_NAMES
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nAll features saved to {output_csv}")

    # Also save per-source summaries
    summary_path = os.path.join(base_dir, "data", "baselines", "feature_summary.csv")
    with open(summary_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Source", "Metric", "Mean", "Std"])
        for label in dirs.keys():
            group = [r for r in all_results if r["source"] == label]
            if not group:
                continue
            for fn in FEATURE_NAMES:
                values = [r[fn] for r in group]
                writer.writerow([label, fn, np.mean(values), np.std(values)])
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
