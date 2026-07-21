import os
import sys
import torch
import pickle
import pretty_midi
from collections import defaultdict
from music21 import stream, note
import torch.nn as nn
import random
import numpy as np
from datetime import datetime
# ======================
# ARGUMENTS
# ======================
# Uso: python script.py <input_folder> <seed> <sigma> <N>
input_folder = sys.argv[1]
seed = int(sys.argv[2])
sigma = float(sys.argv[3])
N = int(sys.argv[4])

date = datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================
# EXPERIMENT ID
# ======================
input_name = os.path.basename(os.path.normpath(input_folder))
exp_name = f"{input_name}_seed{seed}_sigma{sigma}_N{N}"
base_output_folder = "./generated_midis"
#Output folder + date
output_folder = os.path.join(base_output_folder, exp_name + "_" + date)
os.makedirs(output_folder, exist_ok=True)

# ======================
# SET SEEDS
# ======================
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ======================
# MODEL
# ======================
class CVAE_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, vocab_size, embedding_dim,
                 num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                    batch_first=True, dropout=dropout)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        x = self.encoder_embedding(x)
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def generate(self, z, max_len, start_token_id, eos_token_id):
        batch_size = z.size(0)
        hidden_state = torch.tanh(self.latent_to_hidden(z))
        hidden = hidden_state.view(self.decoder_lstm.num_layers, batch_size, self.hidden_dim)
        cell = torch.zeros_like(hidden)
        inputs = torch.full((batch_size,), start_token_id, dtype=torch.long)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(inputs).unsqueeze(1)
            out, (hidden, cell) = self.decoder_lstm(emb, (hidden, cell))
            logits = self.output_layer(out.squeeze(1))
            probs = torch.softmax(logits, dim=-1)
            inputs = torch.multinomial(probs, 1).squeeze(1)
            generated.append(inputs)
            if eos_token_id is not None and (inputs == eos_token_id).all():
                break
        return torch.stack(generated, dim=1)

# ======================
# LOAD MODEL & DICTS
# ======================
with open("Models/token_to_id_train.pkl", "rb") as f:
    token_to_id = pickle.load(f)
with open("Models/id_to_token_train.pkl", "rb") as f:
    id_to_token = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#vocab_size = vocab_size = 456
vocab_size = 83
model = CVAE_LSTM(input_dim=32, hidden_dim=64, embedding_dim=64,
                  latent_dim=64, num_layers=1, dropout=0.0,
                  vocab_size=vocab_size).to(device)

state = torch.load("Models/scl_fulltrained.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# ======================
# FAST REWARD
# ======================
def fast_reward(tokens, escala_mod12, root_note):
    tonal = 0
    tension = 0
    jump = 0
    prev_root = None

    for tok in tokens:
        chord = id_to_token[tok]
        if chord in ['<sos>', '<eos>', '<pad>']:
            continue

        notas = [root_note + int(x) for x in chord.split('_')]

        if any(n % 12 not in escala_mod12 for n in notas):
            tonal += 1

        if len(notas) < 4:
            tension += 1

        root = min(notas)
        if prev_root is not None:
            jump += max(0, abs(root - prev_root) - 7)
        prev_root = root

    return tonal + tension + 0.1 * jump



# ======================
# AUX FUNCTIONS
# ======================
def midi_to_chords(midi_path, tol=0.5, min_notes=2):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    by_time = defaultdict(list)
    for n in notes:
        key = round(n.start / tol) * tol
        by_time[key].append(n.pitch)
    chords = []
    for t in sorted(by_time):
        if len(by_time[t]) >= min_notes:
            chords.append(sorted({p % 12 for p in by_time[t]}))
    return chords

def progression_to_tokens(prog):
    out = []
    for ch in prog:
        tok = "_".join(map(str, ch))
        out.append(token_to_id.get(tok, token_to_id["<pad>"]))
    return out

def build_midi_from_chords(chords, output_path, base_pitch=60, duration=0.5):
    pm_out = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    time = 0.0

    for chord_str in chords:
        if chord_str in ["<sos>", "<eos>", "<pad>"]:
            continue
        notas = [int(n) for n in chord_str.split("_") if n.isdigit()]
        for n in notas:
            piano.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=base_pitch + n,
                    start=time,
                    end=time + duration
                )
            )
        time += duration

    pm_out.instruments.append(piano)
    pm_out.write(output_path)

def get_global_root(tokens):
    for tok in tokens:
        chord = id_to_token[tok]
        if chord not in ['<sos>', '<eos>', '<pad>']:
            return min(int(x) for x in chord.split('_'))
    return 0

def get_scale_from_tokens(tokens, root_note):
    scale = set()
    for tok in tokens:
        chord = id_to_token[tok]
        if chord not in ['<sos>', '<eos>', '<pad>']:
            for x in chord.split('_'):
                scale.add((root_note + int(x)) % 12)
    return scale

def detectar_escala(notas):
    s = stream.Stream()
    for n in notas:
        s.append(note.Note(n))
    k = s.analyze("key")
    return sorted({p.pitchClass for p in k.getScale().getPitches()})

def loss_coherencia(tokens, escala):
    notas = []
    for tok in tokens:
        for p in tok.split("_"):
            if p.isdigit():
                notas.append(int(p) % 12)
    if not notas:
        return 1.0
    en_escala = sum(1 for n in notas if n in escala)
    return 1 - en_escala / len(notas)

# ======================
# PROCESS ALL MIDIS IN FOLDER
# ======================
midi_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mid")]
print(f"[ℹ️] {len(midi_files)} MIDI files in {input_folder}")

for archivo in midi_files:
    input_path = os.path.join(input_folder, archivo)
    try:
        prog = midi_to_chords(input_path)
        if not prog:
            print(f"No chords found in {archivo}, skipping.")
            continue

        x_tokens = progression_to_tokens(prog)
        x = torch.tensor(x_tokens, dtype=torch.long).unsqueeze(0).to(device)

        root_note = get_global_root(x_tokens)
        escala = get_scale_from_tokens(x_tokens, root_note)
        input_tokens = [id_to_token[t] for t in x_tokens if id_to_token[t] not in ["<sos>", "<eos>", "<pad>"]]

        best_score = float("inf")
        best_tokens = input_tokens

        with torch.no_grad():
            mu, logvar = model.encode(x)
            std = torch.exp(0.5 * logvar)

            for _ in range(N):
                z = mu + sigma * std * torch.randn_like(std)
                out = model.generate(
                    z, max_len=len(x_tokens),
                    start_token_id=token_to_id["<sos>"],
                    eos_token_id=token_to_id["<eos>"]
                )[0]

                # tokens
                tokens_ids = [t.item() for t in out if id_to_token[t.item()] not in ["<sos>", "<eos>", "<pad>"]]
                if tokens_ids == x_tokens:
                    continue

                # FAST_REWARD
                score = fast_reward(tokens_ids, escala, root_note)

                if score < best_score:
                    best_score = score
                    best_tokens = [id_to_token[t] for t in tokens_ids]

        # SAVE MIDI
        output_path = os.path.join(
            output_folder,
            f"{os.path.splitext(archivo)[0]}_SCL_seed{seed}_sigma{sigma}_N{N}_{date}.mid"
        )
        build_midi_from_chords(best_tokens, output_path)
        print(f"[✅] {archivo} → {output_path} (score={best_score:.4f})")

    except Exception as e:
        print(f"[❌] Error in {archivo}: {e}")
