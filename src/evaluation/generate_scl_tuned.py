"""
Generate SCL Tuned outputs (tau=1.26, N=19) for relatedness computation.
Loads the full-trained checkpoint and generates enriched progressions.
"""
import os
import sys
import torch
import pickle
import numpy as np
import pretty_midi
import torch.nn as nn
import torch.nn.functional as F


class CVAE_LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, embedding_dim=64, latent_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        x_emb = self.encoder_embedding(x)
        _, (h_n, _) = self.encoder_lstm(x_emb)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def decode_greedy(self, z, max_len, sos_token, eos_token):
        batch_size = z.size(0)
        h0 = self.latent_to_hidden(z)
        h0 = h0.view(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros_like(h0)
        input_token = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=z.device)
        hidden, cell = h0, c0
        generated = []
        for _ in range(max_len):
            emb = self.embedding(input_token)
            lstm_out, (hidden, cell) = self.decoder_lstm(emb, (hidden, cell))
            logits = self.output_layer(lstm_out.squeeze(1))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            generated.append(next_token.item())
            input_token = next_token
        return generated

    def decode_sample(self, z, max_len, sos_token, eos_token, temperature=1.0):
        batch_size = z.size(0)
        h0 = self.latent_to_hidden(z)
        h0 = h0.view(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros_like(h0)
        input_token = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=z.device)
        hidden, cell = h0, c0
        generated = []
        for _ in range(max_len):
            emb = self.embedding(input_token)
            lstm_out, (hidden, cell) = self.decoder_lstm(emb, (hidden, cell))
            logits = self.output_layer(lstm_out.squeeze(1)) / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token.item())
            input_token = next_token
        return generated


def chord_to_pcs(token_str):
    try:
        return {int(x) for x in token_str.split("_")}
    except:
        return set()


def compute_losses(tokens, input_tokens, id_to_token):
    """Compute coherence, tension, movement losses for candidate selection."""
    if len(tokens) != len(input_tokens):
        return 9999

    # Infer tonal scale from input
    all_pcs = set()
    for tid in input_tokens:
        ts = id_to_token.get(tid, "")
        all_pcs.update(chord_to_pcs(ts))
    scale = set(all_pcs)  # simplified: use input pitch classes as scale reference

    coherence = 0.0
    tension = 0.0
    movement = 0.0

    prev_root = None
    for tid in tokens:
        ts = id_to_token.get(tid, "")
        pcs = chord_to_pcs(ts)
        if not pcs:
            continue

        # Tension: penalize chords with <4 notes
        if len(pcs) < 4:
            tension += 1.0

        # Coherence: penalize notes outside input scale
        outside = sum(1 for pc in pcs if pc not in scale)
        coherence += outside / max(len(pcs), 1)

        # Movement: penalize large root jumps
        root = min(pcs)
        if prev_root is not None:
            jump = abs(root - prev_root)
            if jump > 7:
                movement += (jump - 7)
        prev_root = root

    n = len(tokens)
    return 2.0 * coherence / n + 0.1 * tension / n + 0.3 * movement / max(n - 1, 1)


def midi_to_tokens(midi_path, token_to_id):
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted(midi.instruments[0].notes, key=lambda x: x.start)
    if not notes:
        return []
    tokens = []
    current_chord = set()
    current_time = notes[0].start
    for note in notes:
        if note.start - current_time < 0.05:
            current_chord.add(note.pitch % 12)
        else:
            if current_chord:
                cs = "_".join(str(pc) for pc in sorted(current_chord))
                if cs in token_to_id:
                    tokens.append(token_to_id[cs])
            current_time = note.start
            current_chord = {note.pitch % 12}
    if current_chord:
        cs = "_".join(str(pc) for pc in sorted(current_chord))
        if cs in token_to_id:
            tokens.append(token_to_id[cs])
    return tokens


def tokens_to_midi(tokens, id_to_token, output_path):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    t = 0.0
    for tid in tokens:
        ts = id_to_token.get(tid, "")
        if ts in ("<pad>", "<sos>", "<eos>", ""):
            continue
        try:
            pcs = [int(x) for x in ts.split("_")]
        except ValueError:
            continue
        for pc in pcs:
            piano.notes.append(pretty_midi.Note(velocity=80, pitch=60 + pc, start=t, end=t + 2.0))
        t += 2.0
    midi.instruments.append(piano)
    midi.write(output_path)


def main():
    base = "/home/pepebeats/Semi-Supervised-CVAE-LSTM/ICTAI 2026"
    vocab_path = os.path.join(base, "src", "checkpoints", "token_to_id_train.pkl")
    model_path = os.path.join(base, "src", "checkpoints", "cvae_lstm_model.pth")
    input_dir = os.path.join(base, "data", "midi", "input")
    output_dir = os.path.join(base, "data", "midi", "scl_tuned_output")

    os.makedirs(output_dir, exist_ok=True)

    # Load vocab
    with open(vocab_path, "rb") as f:
        token_to_id = pickle.load(f)
    id_to_token = {v: k for k, v in token_to_id.items()}
    vocab_size = len(token_to_id)
    sos_token = token_to_id.get("<sos>", 1)
    print(f"Vocab: {vocab_size} tokens")

    # Load model
    model = CVAE_LSTM(vocab_size=vocab_size)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("Model loaded")

    # Parameters
    tau = 1.26
    N = 19
    temperature = 1.0
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".mid")])

    for fname in input_files:
        prefix = fname.split("SIMPLE")[0]
        input_path = os.path.join(input_dir, fname)
        input_tokens = midi_to_tokens(input_path, token_to_id)
        if not input_tokens:
            print(f"  {prefix}: no tokens found, skipping")
            continue
        max_len = len(input_tokens)

        # Encode
        x = torch.tensor([input_tokens], dtype=torch.long)
        with torch.no_grad():
            mu, logvar = model.encode(x)
            std = torch.exp(0.5 * logvar)

        # Generate N candidates
        best_loss = float("inf")
        best_tokens = None
        n_valid = 0

        for _ in range(N):
            z = mu + tau * std * torch.randn_like(std)
            with torch.no_grad():
                gen = model.decode_sample(z, max_len, sos_token, 0, temperature)
            gen_tokens = gen[:max_len]
            if gen_tokens == input_tokens:
                continue  # skip identical
            loss = compute_losses(gen_tokens, input_tokens, id_to_token)
            n_valid += 1
            if loss < best_loss:
                best_loss = loss
                best_tokens = gen_tokens

        if best_tokens is None:
            print(f"  {prefix}: no valid candidates (all identical to input)")
            continue

        out_path = os.path.join(output_dir, f"{prefix}SCL_TUNED.mid")
        tokens_to_midi(best_tokens, id_to_token, out_path)
        print(f"  {prefix}: saved ({n_valid}/{N} valid, loss={best_loss:.3f})")

    print(f"\nOutputs saved to {output_dir}")
    print("Now run: python3 src/evaluation/relatedness.py (after adjusting paths)")


if __name__ == "__main__":
    main()
