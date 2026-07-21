import os
import re
import torch
import pretty_midi
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import torch.nn as nn
from collections import defaultdict
from music21 import chord
import difflib
from music21 import stream, note

class CVAE_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, vocab_size, embedding_dim, num_layers=2, dropout=0.2):
        super(CVAE_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # Encoder
        self.encoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        x = self.encoder_embedding(x)  # [batch, seq_len, emb_dim]
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_seq, teacher_forcing_ratio=0.4):
        batch_size, seq_len = y_seq.size()
        embedding = self.embedding

        hidden_state = torch.tanh(self.latent_to_hidden(z))
        hidden = hidden_state.view(self.decoder_lstm.num_layers, batch_size, self.hidden_dim)
        cell = torch.zeros_like(hidden).to(hidden.device)

        inputs = y_seq[:, 0]  # token <sos>
        outputs = []

        for t in range(1, seq_len):
            input_embed = embedding(inputs).unsqueeze(1)
            output, (hidden, cell) = self.decoder_lstm(input_embed, (hidden, cell))
            output_logits = self.output_layer(output.squeeze(1))  # [batch, vocab_size]
            outputs.append(output_logits)

            teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = output_logits.argmax(1)
            inputs = y_seq[:, t] if teacher_force else top1

        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len-1, vocab_size]
        return outputs


    def forward(self, x, y_seq=None, teacher_forcing_ratio=0.8):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if y_seq is not None:
            y_hat = self.decode(z, y_seq, teacher_forcing_ratio)
            return y_hat, mu, logvar
        else:
            # Modo generación sin y_seq
            y_hat = self.generate(z, max_len=10)  # max_len puede ser parámetro externo
            return y_hat, mu, logvar

    # Método para generar secuencias a partir del vector latente z.
    def generate(self, z, max_len=10, start_token_id=None, eos_token_id=None):
        batch_size = z.size(0)
        hidden_state = torch.tanh(self.latent_to_hidden(z))
        hidden = hidden_state.view(self.decoder_lstm.num_layers, batch_size, self.hidden_dim)
        cell = torch.zeros_like(hidden).to(hidden.device)

        inputs = torch.full((batch_size,), start_token_id, dtype=torch.long).to(z.device)
        generated_tokens = []

        for _ in range(max_len):
            input_embed = self.embedding(inputs).unsqueeze(1)
            output, (hidden, cell) = self.decoder_lstm(input_embed, (hidden, cell))
            output_logits = self.output_layer(output.squeeze(1))  # [batch, vocab_size]

            probs = torch.softmax(output_logits, dim=-1)
            inputs = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated_tokens.append(inputs)

            if eos_token_id is not None:
                if (inputs == eos_token_id).all():
                    break

        generated_tokens = torch.stack(generated_tokens, dim=1)  # [batch, seq_len]
        return generated_tokens


with open("/token_to_id_train.pkl", "rb") as f:
    token_to_id_train = pickle.load(f)

with open("//id_to_token_train.pkl", "rb") as f:
    id_to_token_train = pickle.load(f)

TOLERANCE = 0.1
MIN_NOTES = 2
BATCH_SIZE = 8


def midi_to_chords_train(midi_path, tolerance=TOLERANCE, min_notes=MIN_NOTES):
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        all_notes.extend(instrument.notes)

    notes_by_time = defaultdict(list)
    for note in all_notes:
        time_key = round(note.start / tolerance) * tolerance
        notes_by_time[time_key].append(note.pitch)

    chords = []
    for time in sorted(notes_by_time.keys()):
        note_group = notes_by_time[time]
        if len(note_group) < min_notes:
            continue
        try:
            m21_chord = chord.Chord(note_group)
            pc_set = sorted(set(n % 12 for n in note_group))
        except Exception:
            pc_set = sorted(set(n % 12 for n in note_group))
        chords.append(pc_set)
    return chords


def chord_to_token(chord):
    return '_'.join(map(str, sorted(chord)))

def fallback_token_str(chord_str, token_to_id):
    matches = difflib.get_close_matches(chord_str, token_to_id.keys(), n=1)
    return token_to_id[matches[0]] if matches else token_to_id.get('<pad>', 0)

def progression_to_token_seq(prog, token_to_id):
    tokens = []
    for chord in prog:
        token = chord_to_token(chord)
        if token in token_to_id:
            tokens.append(token_to_id[token])
        else:
            tokens.append(fallback_token_str(token, token_to_id))
    return tokens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 456
model = CVAE_LSTM(
    input_dim=32, 
    hidden_dim=64,
    embedding_dim=64,
    latent_dim=64,
    num_layers=1, 
    dropout=0.3,
    vocab_size=vocab_size,
).to(device)

model.load_state_dict(torch.load("/cvae_lstm_model.pth", map_location="cpu"))
model.eval()

def loss_coherencia_tonal_soft(y_hat_step, root_note, escala_mod12, id_to_token):
    probs = F.softmax(y_hat_step, dim=-1)
    fuera_escala = torch.zeros(probs.size(-1), device=probs.device)
    
    for idx in range(probs.size(-1)):
        chord = id_to_token.get(idx, None)
        if chord is None:
            fuera_escala[idx] = 0
            continue

        if chord in ['<sos>', '<eos>', '<pad>']:
            fuera_escala[idx] = 0
        else:
            try:
                notas = [root_note + int(x) for x in chord.split('_')]
                count_fuera = sum(1 for n in notas if n % 12 not in escala_mod12)
                fuera_escala[idx] = count_fuera / len(notas)
            except:
                fuera_escala[idx] = 0

    loss = torch.sum(probs * fuera_escala)
    return loss

def detectar_escala(notas_midi):
    s = stream.Stream()
    for n in notas_midi:
        s.append(note.Note(n))
    k = s.analyze('key')
    escala = k.getScale().getPitches()
    escala_mod12 = sorted({p.pitchClass for p in escala})
    return escala_mod12

def generar_mejor_progresion(model, x_input, token_to_id, id_to_token, escala_mod12, root_note, N=50, max_len=8):
    model.eval()
    mejores_tokens = None
    mejor_loss = float('inf')
    longitud_objetivo = max_len

    with torch.no_grad():
        mu, logvar = model.encode(x_input)
        for _ in range(N):
            z = model.reparameterize(mu, logvar)
            salida = model.generate(
                z,
                max_len=max_len,
                start_token_id=token_to_id['<sos>'],
                eos_token_id=token_to_id['<eos>']
            )[0]

            salida_list = salida.cpu().tolist()
        
            salida_chords = [
                id_to_token[token] 
                for token in salida_list 
                if id_to_token[token] not in ['<sos>', '<eos>', '<pad>']
            ]

            if len(salida_chords) == longitud_objetivo:
                logits = F.one_hot(salida, num_classes=len(token_to_id)).float()
                loss = sum(
                    loss_coherencia_tonal_soft(y_hat_step, root_note, escala_mod12, id_to_token)
                    for y_hat_step in logits
                )

                if loss < mejor_loss:
                    mejor_loss = loss
                    mejores_tokens = salida

    if mejores_tokens is None:
        mejor_loss = float('inf')
        for _ in range(N):
            z = model.reparameterize(mu, logvar)
            salida = model.generate(
                z,
                max_len=max_len,
                start_token_id=token_to_id['<sos>'],
                eos_token_id=token_to_id['<eos>']
            )[0]

            logits = F.one_hot(salida, num_classes=len(token_to_id)).float()
            loss = sum(
                loss_coherencia_tonal_soft(y_hat_step, root_note, escala_mod12, id_to_token)
                for y_hat_step in logits
            )

            if loss < mejor_loss:
                mejor_loss = loss
                mejores_tokens = salida

    return [id_to_token[token.item()] for token in mejores_tokens]

def chords_to_notes(chord_str):
    return [int(n) for n in chord_str.split('_') if n.isdigit()]

def build_midi_from_chords(generated_chords, reference_midi_path, output_path):
    original_midi = pretty_midi.PrettyMIDI(reference_midi_path)
    original_notes = [n.pitch for inst in original_midi.instruments if not inst.is_drum for n in inst.notes]
    if not original_notes:
        base_pitch = 60 
    else:
        base_pitch = min(original_notes)
    durations = [n.end - n.start for inst in original_midi.instruments if not inst.is_drum for n in inst.notes]
    chord_duration = sum(durations) / len(durations) if durations else (original_midi.get_end_time() / len(generated_chords))

    new_midi = pretty_midi.PrettyMIDI(initial_tempo=original_midi.get_tempo_changes()[1][0])
    piano = pretty_midi.Instrument(program=0)

    time = 0.0
    for chord_str in generated_chords:
        if chord_str in ['<sos>', '<eos>', '<pad>']:
            continue
        notes = [int(n) for n in chord_str.split('_') if n.isdigit()]
        for interval in notes:
            pitch = base_pitch + interval
            if pitch < 0:
                pitch = 0
            elif pitch > 127:
                pitch = 127
            note_obj = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=time,
                end=time + chord_duration
            )
            piano.notes.append(note_obj)
        time += chord_duration

    new_midi.instruments.append(piano)
    new_midi.write(output_path)



from music21 import chord

def token_to_chord_symbol(token, base_pitch=60):
    try:
        notas = [base_pitch + int(n) for n in token.split('_') if n.isdigit()]
        if not notas:
            return token
        c = chord.Chord(notas)
        return c.pitchedCommonName  # Ej: 'C minor triad'
    except Exception:
        return token

def loss_coherencia_tonal_tokens(tokens, escala_mod12, root_note):
    """
    Evalúa coherencia tonal directamente a partir de tokens decodificados.
    tokens: lista de strings tipo '0_4_7'
    """
    notas_midi = []
    for tok in tokens:
        for p in tok.split('_'):
            if p.isdigit():
                notas_midi.append(int(p) % 12)

    # Conteo: cuántas notas están en la escala detectada
    en_escala = sum(1 for n in notas_midi if n in escala_mod12)
    total = len(notas_midi)

    if total == 0:
        return float("inf")  # penalizar vacío
    else:
        return 1 - (en_escala / total)  # mientras más bajo, mejor

def generar_mejores_variantes(model, x_input, token_to_id, id_to_token, escala_mod12, root_note,
                              num_generadas=50, num_mejores=10, max_len=5):
    model.eval()
    variantes = []

    with torch.no_grad():
        mu, logvar = model.encode(x_input)

        for _ in range(num_generadas):
            z = model.reparameterize(mu, logvar)
            salida = model.generate(
                z,
                max_len=max_len,
                start_token_id=token_to_id['<sos>'],
                eos_token_id=token_to_id['<eos>']
            )[0]

            salida_chords = [
                id_to_token[token.item()]
                for token in salida
                if id_to_token[token.item()] not in ['<sos>', '<eos>', '<pad>']
            ]

            # Calcular coherencia tonal
            loss = loss_coherencia_tonal_tokens(salida_chords, escala_mod12, root_note)
            variantes.append((salida_chords, loss))

    # Ordenar por coherencia tonal (menor loss = mejor)
    variantes.sort(key=lambda x: x[1])

    # Devolver solo las mejores N
    return variantes[:num_mejores]

# Example (C minor - G major - C minor - F minor - C minor)
entrada_manual = [
    [0, 3, 7],   # C minor
    [7, 11, 2],  # G major
    [0, 3, 7],   # C minor
    [5, 8, 0],   # F minor
    [0, 3, 7]    # C minor
]

entrada_tokens = progression_to_token_seq(entrada_manual, token_to_id_train)
x_input = torch.tensor(entrada_tokens, dtype=torch.long).unsqueeze(0).to(device)

notas_midi = [
    60 + int(p)
    for tok in entrada_tokens
    for p in id_to_token_train[tok].split('_') if p.isdigit()
]
escala_mod12 = detectar_escala(notas_midi)
root_note = min(notas_midi) % 12

mejores_variantes = generar_mejores_variantes(
    model, x_input,
    token_to_id=token_to_id_train,
    id_to_token=id_to_token_train,
    escala_mod12=escala_mod12,
    root_note=root_note,
    num_generadas=50,
    num_mejores=10,
    max_len=len(entrada_tokens)
)
