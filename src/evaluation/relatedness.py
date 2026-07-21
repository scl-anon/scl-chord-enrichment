"""
Relatedness metrics: input -> output chord progression similarity.
Measures how closely the generated progression relates to the input.
"""
import os
import sys
import csv
import numpy as np
import pretty_midi
from collections import defaultdict

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def midi_to_chord_sequence(midi_path, threshold=0.05):
    """
    Parse MIDI file into sequence of pitch class sets (chords).
    Groups notes within `threshold` seconds into chords.
    """
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
                chords.append(frozenset(current_chord))
            current_time = note.start
            current_chord = {note.pitch % 12}

    if current_chord:
        chords.append(frozenset(current_chord))

    return chords


def chord_to_str(chord):
    """Convert frozenset of pitch classes to underscore-separated string."""
    return "_".join(str(p) for p in sorted(chord))


def chord_overlap_ratio(seq_a, seq_b):
    """
    Fraction of pitch classes in input chord preserved in output chord.
    Averaged over aligned positions (up to min length).
    """
    if not seq_a or not seq_b:
        return 0.0
    ratios = []
    for ca, cb in zip(seq_a, seq_b):
        if not ca:
            ratios.append(0.0)
        else:
            overlap = len(ca & cb) / len(ca)
            ratios.append(overlap)
    return np.mean(ratios) if ratios else 0.0


def chord_jaccard_similarity(seq_a, seq_b):
    """Jaccard similarity between aligned chord positions."""
    if not seq_a or not seq_b:
        return 0.0
    sims = []
    for ca, cb in zip(seq_a, seq_b):
        union = len(ca | cb)
        if union == 0:
            sims.append(1.0)
        else:
            sims.append(len(ca & cb) / union)
    return np.mean(sims) if sims else 0.0


def chord_edit_distance(seq_a, seq_b):
    """
    Levenshtein edit distance between two chord sequences,
    normalized by max sequence length.
    """
    m, n = len(seq_a), len(seq_b)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[m][n] / max(m, n)


def chord_length_ratio(seq_a, seq_b):
    """Ratio of sequence lengths: min/max."""
    if not seq_a or not seq_b:
        return 0.0
    return min(len(seq_a), len(seq_b)) / max(len(seq_a), len(seq_b))


def tonal_center(pitch_classes):
    """Estimate tonal center using Krumhansl-Kessler key profiles."""
    if not pitch_classes:
        return None
    pc_counts = np.zeros(12)
    for pc in pitch_classes:
        pc_counts[pc] += 1

    # Krumhansl-Kessler major key profiles
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    best_corr = -999
    best_key = None
    for key in range(12):
        for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
            shifted = np.roll(profile, key)
            corr = np.corrcoef(pc_counts, shifted)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_key = (key, mode)

    return best_key


def tonal_consistency(seq_a, seq_b):
    """
    Measure tonal center difference between two sequences.
    Returns 1.0 if same key, 0.0 if maximally different (tritone).
    """
    all_pcs_a = set()
    all_pcs_b = set()
    for ch in seq_a:
        all_pcs_a.update(ch)
    for ch in seq_b:
        all_pcs_b.update(ch)

    key_a = tonal_center(all_pcs_a)
    key_b = tonal_center(all_pcs_b)

    if key_a is None or key_b is None:
        return 0.5

    pitch_a = key_a[0]
    # Circular distance: max is 6 (tritone)
    distance = min((pitch_a - key_b[0]) % 12, (key_b[0] - pitch_a) % 12)
    # Same mode bonus
    if key_a[1] == key_b[1]:
        distance = max(0, distance - 1)

    return 1.0 - (distance / 6.0)


def compute_all_relatedness(input_midi, output_midi):
    """Compute all relatedness metrics between input and output MIDI files."""
    seq_in = midi_to_chord_sequence(input_midi)
    seq_out = midi_to_chord_sequence(output_midi)

    return {
        "chord_overlap_ratio": chord_overlap_ratio(seq_in, seq_out),
        "chord_jaccard": chord_jaccard_similarity(seq_in, seq_out),
        "chord_edit_distance": chord_edit_distance(seq_in, seq_out),
        "length_ratio": chord_length_ratio(seq_in, seq_out),
        "tonal_consistency": tonal_consistency(seq_in, seq_out),
    }


def paired_comparison_score(seq_a, seq_b):
    """
    Weighted relatedness score combining overlap, edit distance, and tonality.
    Higher = more related. Range roughly [0, 1].
    """
    metrics = {}
    metrics["chord_overlap_ratio"] = chord_overlap_ratio(seq_a, seq_b)
    metrics["chord_jaccard"] = chord_jaccard_similarity(seq_a, seq_b)
    metrics["chord_edit_distance"] = chord_edit_distance(seq_a, seq_b)
    metrics["length_ratio"] = chord_length_ratio(seq_a, seq_b)
    metrics["tonal_consistency"] = tonal_consistency(seq_a, seq_b)

    # Composite: equally weighted average of all four metrics + length ratio
    composite = (
        0.2 * metrics["chord_overlap_ratio"]
        + 0.2 * metrics["chord_jaccard"]
        + 0.2 * (1.0 - metrics["chord_edit_distance"])
        + 0.2 * metrics["length_ratio"]
        + 0.2 * metrics["tonal_consistency"]
    )

    metrics["relatedness_composite"] = composite
    return metrics


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(base_dir, "data", "midi", "input")
    scl_dir = os.path.join(base_dir, "data", "midi", "scl_output")
    human_dir = os.path.join(base_dir, "data", "midi", "human_output")
    output_csv = os.path.join(base_dir, "data", "features", "relatedness", "relatedness_results.csv")

    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mid")])

    results = []
    print(f"{'Pair':<20} {'Condition':<10} {'Overlap':>8} {'Jaccard':>8} {'EditDist':>8} {'LenRatio':>8} {'Tonal':>8} {'Composite':>8}")
    print("-" * 90)

    for f in input_files:
        # Extract number prefix (e.g., "00001" from "00001SIMPLE_OG.mid")
        prefix = f.split("SIMPLE")[0]
        input_path = os.path.join(input_dir, f)
        scl_path = os.path.join(scl_dir, f"{prefix}REARM_MIO.mid")
        human_path = os.path.join(human_dir, f"{prefix}REARM_OG.mid")

        for condition, output_path in [("SCL", scl_path), ("Human", human_path)]:
            if not os.path.exists(output_path):
                print(f"  WARNING: {output_path} not found, skipping {condition}")
                continue

            metrics = paired_comparison_score(
                midi_to_chord_sequence(input_path),
                midi_to_chord_sequence(output_path),
            )
            metrics["pair"] = prefix
            metrics["condition"] = condition
            results.append(metrics)

            print(
                f"{prefix:<20} {condition:<10} "
                f"{metrics['chord_overlap_ratio']:>8.3f} "
                f"{metrics['chord_jaccard']:>8.3f} "
                f"{metrics['chord_edit_distance']:>8.3f} "
                f"{metrics['length_ratio']:>8.3f} "
                f"{metrics['tonal_consistency']:>8.3f} "
                f"{metrics['relatedness_composite']:>8.3f}"
            )

    # Summary statistics
    print("\n--- Summary ---")
    for condition in ["SCL", "Human"]:
        cond_results = [r for r in results if r["condition"] == condition]
        if cond_results:
            avg_overlap = np.mean([r["chord_overlap_ratio"] for r in cond_results])
            avg_jaccard = np.mean([r["chord_jaccard"] for r in cond_results])
            avg_edit = np.mean([r["chord_edit_distance"] for r in cond_results])
            avg_len = np.mean([r["length_ratio"] for r in cond_results])
            avg_tonal = np.mean([r["tonal_consistency"] for r in cond_results])
            avg_composite = np.mean([r["relatedness_composite"] for r in cond_results])
            print(
                f"  {condition}: Overlap={avg_overlap:.3f}, Jaccard={avg_jaccard:.3f}, "
                f"EditDist={avg_edit:.3f}, LenRatio={avg_len:.3f}, "
                f"Tonal={avg_tonal:.3f}, Composite={avg_composite:.3f}"
            )

    # Save to CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "pair", "condition", "chord_overlap_ratio", "chord_jaccard",
            "chord_edit_distance", "length_ratio", "tonal_consistency",
            "relatedness_composite",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
