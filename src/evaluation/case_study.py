"""
Case study: qualitative analysis of specific chord transformations.
Analyzes what SCL does to chord progressions compared to human reharmonizations.
"""
import os
import sys
import csv
import pretty_midi
from collections import defaultdict


def midi_to_chord_list(midi_path, threshold=0.05):
    """Parse MIDI into list of pitch class sets (as sorted tuples) with labels."""
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


def chord_to_name(pcs):
    """Convert pitch class set to human-readable chord name (approximate)."""
    if not pcs:
        return "?"

    # Build pitch class name map
    pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    sorted_pcs = sorted(pcs)
    root = sorted_pcs[0]
    intervals = tuple((pc - root) % 12 for pc in sorted_pcs[1:])

    # Common chord patterns (root, intervals, name)
    patterns = {
        (4, 7): "",
        (3, 7): "m",
        (3, 6): "dim",
        (4, 8): "aug",
        (4, 7, 10): "7",
        (3, 7, 10): "m7",
        (4, 7, 11): "maj7",
        (3, 6, 10): "m7b5",
        (3, 6, 9): "dim7",
        (4, 8, 10): "aug7",
        (4, 7, 10, 14): "9",
        (3, 7, 10, 14): "m9",
        (4, 7, 11, 14): "maj9",
    }

    name = patterns.get(intervals, "?")
    return f"{pc_names[root]}{name}"


def describe_chord(pcs):
    """Human-readable description of chord structure."""
    name = chord_to_name(pcs)
    notes = len(pcs)
    desc = []
    if notes >= 4:
        desc.append("extended (4+ notes)")
    elif notes == 3:
        desc.append("triad")
    elif notes == 2:
        desc.append("dyad")
    else:
        desc.append("single note")

    # Check for tritone
    intervals = []
    sorted_pcs = sorted(pcs)
    for i in range(len(sorted_pcs)):
        for j in range(i + 1, len(sorted_pcs)):
            interval = (sorted_pcs[j] - sorted_pcs[i]) % 12
            intervals.append(interval)
    if 6 in intervals:
        desc.append("contains tritone")

    return name, ", ".join(desc)


def analyze_transformations(input_pcs, output_pcs):
    """Analyze what changed from input chord to output chord."""
    added = output_pcs - input_pcs
    removed = input_pcs - output_pcs
    kept = input_pcs & output_pcs

    transformations = []
    if added:
        transformations.append(f"added: {sorted(added)}")
    if removed:
        transformations.append(f"removed: {sorted(removed)}")
    if len(output_pcs) > len(input_pcs):
        transformations.append("increased density")
    elif len(output_pcs) < len(input_pcs):
        transformations.append("reduced density")

    # Check if root changed
    if input_pcs and output_pcs:
        input_root = min(input_pcs)
        output_root = min(output_pcs)
        if input_root != output_root:
            diff = (output_root - input_root) % 12
            transformations.append(f"root shifted by {diff} semitones")

    return transformations


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    case_study_dir = os.path.join(base_dir, "experiments", "case_study")
    output_dir = os.path.join(base_dir, "experiments", "case_study")
    os.makedirs(output_dir, exist_ok=True)

    # ComparacionL has filenames with chord names, making it ideal for case study
    # L files = human reharmonizations (from "L"ow complexity set? Actually RVAE comparisons)
    # M files = model (SCL) outputs
    # Let's match pairs and analyze

    midi_files = sorted([f for f in os.listdir(case_study_dir) if f.endswith(".mid")])

    print("=== Case Study: SCL Chord Transformations ===\n")

    # Group L and M files by number
    l_files = {}
    m_files = {}
    for f in midi_files:
        if f.startswith("L"):
            # Extract ID from filename like "L1) Cmin-Gmaj..."
            parts = f.split(")")
            if len(parts) >= 1:
                fid = parts[0].strip()  # e.g., "L1" or "L10"
                l_files[fid] = f
        elif f.startswith("M"):
            parts = f.split(")")
            if len(parts) >= 1:
                fid = parts[0].strip()
                m_files[fid] = f

    # Match pairs (L1 <-> M1, etc. mapping by context)
    # From the data: L1 input is Cmin-Gmaj-Cmin-Fmin-Cmin (same input for comparison)
    # M1, M2, M3 are different SCL variants of L1's input
    # M22 is a variant, M4, M6 are variants of L3/L4 inputs

    print("SCL-generated progressions and their characteristics:\n")

    for fname in sorted(m_files.values()):
        path = os.path.join(case_study_dir, fname)
        chords = midi_to_chord_list(path)
        print(f"  {fname}: ", end="")
        chord_strs = [chord_to_name(c) for c in chords]
        print(" – ".join(chord_strs))

        # Analyze chord complexity
        n_extended = sum(1 for c in chords if len(c) >= 4)
        n_triads = sum(1 for c in chords if len(c) == 3)
        has_tritone = any(
            any((c[j] - c[i]) % 12 == 6 for i in range(len(c)) for j in range(i + 1, len(c)))
            for c in chords
        )
        print(f"      Length: {len(chords)} chords, {n_triads} triads, {n_extended} extended, "
              f"tritone: {'yes' if has_tritone else 'no'}")
        print()

    # Compare with a specific human reference from ComparacionH
    print("=" * 60)
    print("Comparison: Input progression Cmin-Gmaj-Cmin-Fmin-Cmin\n")

    # The input used for Block 1 comparisons
    input_dir = os.path.join(base_dir, "data", "midi", "input")
    scl_dir = os.path.join(base_dir, "data", "midi", "scl_output")
    human_dir = os.path.join(base_dir, "data", "midi", "human_output")

    for f in sorted(os.listdir(input_dir)):
        prefix = f.split("SIMPLE")[0]
        input_path = os.path.join(input_dir, f)
        scl_path = os.path.join(scl_dir, f"{prefix}REARM_MIO.mid")
        human_path = os.path.join(human_dir, f"{prefix}REARM_OG.mid")

        if not os.path.exists(scl_path):
            continue

        input_chords = midi_to_chord_list(input_path)
        scl_chords = midi_to_chord_list(scl_path)
        human_chords = midi_to_chord_list(human_path) if os.path.exists(human_path) else []

        print(f"  Pair: {prefix}")
        print(f"    Input:  {' – '.join(chord_to_name(c) for c in input_chords)}")
        print(f"    SCL:    {' – '.join(chord_to_name(c) for c in scl_chords)}")
        if human_chords:
            print(f"    Human:  {' – '.join(chord_to_name(c) for c in human_chords)}")

        # Show transformations
        print(f"    Transformations (Input -> SCL):")
        for i, (in_c, out_c) in enumerate(zip(input_chords, scl_chords)):
            if in_c != out_c:
                trans = analyze_transformations(set(in_c), set(out_c))
                name_in = chord_to_name(in_c)
                name_out = chord_to_name(out_c)
                print(f"      Chord {i+1}: {name_in} -> {name_out}: {'; '.join(trans)}")

        if human_chords:
            print(f"    Human divergences from input:")
            for i, (in_c, hum_c) in enumerate(zip(input_chords, human_chords)):
                if in_c != hum_c:
                    trans = analyze_transformations(set(in_c), set(hum_c))
                    name_in = chord_to_name(in_c)
                    name_hum = chord_to_name(hum_c)
                    print(f"      Chord {i+1}: {name_in} -> {name_hum}: {'; '.join(trans)}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
