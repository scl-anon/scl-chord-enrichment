"""
Generate random and mismatched baselines for chord progression evaluation.
Creates MIDI files, extracts jSymbolic features, and compares complexity metrics.
"""
import os
import sys
import csv
import random
import subprocess
import numpy as np
import pickle
import pretty_midi


def load_vocabulary():
    """Load chord token vocabulary."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    vocab_path = os.path.join(base_dir, "src", "checkpoints", "token_to_id_train.pkl")
    with open(vocab_path, "rb") as f:
        token_to_id = pickle.load(f)

    # Get all chord tokens (exclude special tokens like pad, sos, eos)
    chord_tokens = [k for k in token_to_id.keys() if k not in ("<pad>", "<sos>", "<eos>")]
    return chord_tokens


def token_to_midi_notes(token):
    """Convert a chord token like '0_4_7' to list of MIDI pitch values."""
    try:
        pitch_classes = [int(x) for x in token.split("_")]
    except (ValueError, AttributeError):
        return []
    # Use octave 4 (middle C = 60) as base
    return [60 + pc for pc in pitch_classes]


def create_midi_from_chord_tokens(tokens, output_path, tempo=120, chord_duration=2.0):
    """Create a MIDI file from a list of chord tokens."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    time = 0.0
    for token in tokens:
        notes = token_to_midi_notes(token)
        for pitch in notes:
            note = pretty_midi.Note(
                velocity=80, pitch=pitch,
                start=time, end=time + chord_duration
            )
            piano.notes.append(note)
        time += chord_duration

    midi.instruments.append(piano)
    midi.write(output_path)


def count_chords_in_midi(midi_path, threshold=0.05):
    """Count number of chords in a MIDI file."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda x: x.start)
    if not notes:
        return 0

    chords = 1
    current_time = notes[0].start
    for note in notes[1:]:
        if note.start - current_time >= threshold:
            chords += 1
            current_time = note.start
    return chords


def generate_random_baselines(input_dir, output_dir, chord_vocabulary):
    """
    Generate random chord sequences matching the length of each input progression.
    """
    os.makedirs(output_dir, exist_ok=True)
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mid")])

    for f in input_files:
        input_path = os.path.join(input_dir, f)
        n_chords = count_chords_in_midi(input_path)

        # Generate random chord sequence of same length
        random_tokens = random.choices(chord_vocabulary, k=n_chords)

        prefix = f.split("SIMPLE")[0]
        output_path = os.path.join(output_dir, f"{prefix}RANDOM.mid")
        create_midi_from_chord_tokens(random_tokens, output_path)
        print(f"  Created random baseline: {prefix}RANDOM.mid ({n_chords} chords)")

    return output_dir


def generate_mismatched_baselines(input_dir, output_dir, chord_vocabulary):
    """
    Generate mismatched sequences: take real chord progressions from the
    human reharm set and pair them with different inputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all real reharm sequences
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    human_dir = os.path.join(base_dir, "data", "midi", "human_output")
    human_files = sorted([f for f in os.listdir(human_dir) if f.endswith(".mid")])

    if not human_files:
        print("  No human reharm files found for mismatched baseline.")
        return output_dir

    # Extract chord sequences from human reharmonizations
    human_sequences = {}
    for hf in human_files:
        hpath = os.path.join(human_dir, hf)
        midi = pretty_midi.PrettyMIDI(hpath)
        notes = sorted(midi.instruments[0].notes, key=lambda x: x.start)
        chords = []
        current = set()
        t0 = None
        for n in notes:
            if t0 is None:
                t0 = n.start
                current = {n.pitch % 12}
            elif n.start - t0 < 0.05:
                current.add(n.pitch % 12)
            else:
                if current:
                    chords.append(frozenset(current))
                t0 = n.start
                current = {n.pitch % 12}
        if current:
            chords.append(frozenset(current))
        human_sequences[hf] = chords

    # Mismatch: assign human reharm from one progression to input of another
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mid")])
    shuffled_human = list(human_files)
    random.shuffle(shuffled_human)

    for i, f in enumerate(input_files):
        mismatched_human_file = shuffled_human[i % len(shuffled_human)]
        mismatched_path = os.path.join(human_dir, mismatched_human_file)

        prefix = f.split("SIMPLE")[0]
        output_path = os.path.join(output_dir, f"{prefix}MISMATCHED.mid")

        # Copy the mismatched MIDI with renamed prefix
        import shutil
        shutil.copy(mismatched_path, output_path)
        print(f"  Created mismatched baseline: {prefix}MISMATCHED.mid <- {mismatched_human_file}")

    return output_dir


def run_jsymbolic(midi_dir, output_csv):
    """
    Run jSymbolic feature extraction on a directory of MIDI files.
    Requires Java to be installed.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    jar_path = os.path.join(base_dir, "experiments", "jsymbolic", "jSymbolic2.jar")

    if not os.path.exists(jar_path):
        print(f"  WARNING: jSymbolic JAR not found at {jar_path}")
        print(f"  Skipping feature extraction. Please run jSymbolic manually.")
        return False

    abs_midi_dir = os.path.abspath(midi_dir)
    abs_output = os.path.abspath(output_csv)

    # jSymbolic command line: java -jar jSymbolic2.jar -csv <output> <input_dir>
    cmd = [
        "java", "-jar", jar_path,
        "-csv", abs_output,
        abs_midi_dir,
    ]

    print(f"  Running jSymbolic on {abs_midi_dir}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"  Features saved to {abs_output}")
            return True
        else:
            print(f"  jSymbolic error: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("  jSymbolic timed out (120s)")
        return False
    except FileNotFoundError:
        print("  Java not found. Please install Java to run jSymbolic.")
        return False


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(base_dir, "data", "midi", "input")
    random_dir = os.path.join(base_dir, "data", "baselines", "random")
    mismatched_dir = os.path.join(base_dir, "data", "baselines", "mismatched")

    print("Loading chord vocabulary...")
    vocabulary = load_vocabulary()
    print(f"  Loaded {len(vocabulary)} chord types")

    print("\n=== Generating Random Baselines ===")
    generate_random_baselines(input_dir, random_dir, vocabulary)

    print("\n=== Generating Mismatched Baselines ===")
    generate_mismatched_baselines(input_dir, mismatched_dir, vocabulary)

    # Run jSymbolic on baseline directories
    print("\n=== Extracting jSymbolic Features ===")
    random_features_csv = os.path.join(base_dir, "data", "baselines", "random_features.csv")
    mismatched_features_csv = os.path.join(base_dir, "data", "baselines", "mismatched_features.csv")

    run_jsymbolic(random_dir, random_features_csv)
    run_jsymbolic(mismatched_dir, mismatched_features_csv)

    print("\nDone. Baseline MIDIs and features generated.")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
