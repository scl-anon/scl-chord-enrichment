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
from midi2audio import FluidSynth

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

TOLERANCE = 0.1
MIN_NOTES = 2
BATCH_SIZE = 8

def loss_coherencia_tonal_soft(y_hat_step, root_note, escala_mod12, id_to_token):
    probs = F.softmax(y_hat_step, dim=-1)  # [batch, vocab_size]
    fuera_escala = torch.zeros(probs.size(-1), device=probs.device)
    
    for idx in range(probs.size(-1)):
        chord = id_to_token.get(idx, None)
        if chord is None:
            fuera_escala[idx] = 0
            continue

        if chord in ['<sos>', '<eos>', '<pad>']:
            fuera_escala[idx] = 0  # No penalizar tokens especiales
        else:
            try:
                notas = [root_note + int(x) for x in chord.split('_')]
                count_fuera = sum(1 for n in notas if n % 12 not in escala_mod12)
                fuera_escala[idx] = count_fuera / len(notas)
            except:
                fuera_escala[idx] = 0

    loss = torch.sum(probs * fuera_escala)
    return loss



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



# Función auxiliar para obtener escala mod12 desde notas MIDI (sin cambios)
def detectar_escala(notas_midi):
    s = stream.Stream()
    for n in notas_midi:
        s.append(note.Note(n))
    k = s.analyze('key')
    escala = k.getScale().getPitches()
    escala_mod12 = sorted({p.pitchClass for p in escala})
    return escala_mod12


from music21 import chord

# === Helper: token a símbolo de acorde ===
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



# === Nueva función: generar y seleccionar las mejores variantes ===
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

def generar_mejores_variantes_con_longitud(model, x_input, token_to_id, id_to_token,
                                           escala_mod12, root_note, longitud_objetivo,
                                           num_generadas=100, num_mejores=10):
    model.eval()
    variantes = []

    with torch.no_grad():
        mu, logvar = model.encode(x_input)

        for _ in range(num_generadas):
            z = model.reparameterize(mu, logvar)
            salida = model.generate(
                z,
                max_len=longitud_objetivo + 2,  # +2 por <sos> y <eos>
                start_token_id=token_to_id['<sos>'],
                eos_token_id=token_to_id['<eos>']
            )[0]

            salida_chords = [
                id_to_token[token.item()]
                for token in salida
                if id_to_token[token.item()] not in ['<sos>', '<eos>', '<pad>']
            ]

            # Filtrar por longitud exacta
            if len(salida_chords) != longitud_objetivo:
                continue

            loss = loss_coherencia_tonal_tokens(salida_chords, escala_mod12, root_note)
            variantes.append((salida_chords, loss))

    if not variantes:
        print("[⚠️] No se generaron variantes con la longitud exacta. Ajustar num_generadas o tolerancia.")
        return []

    variantes.sort(key=lambda x: x[1])  # ordenar por menor loss
    return variantes[:num_mejores]

def chords_to_notes(chord_str):
    return [int(n) for n in chord_str.split('_') if n.isdigit()]



def calcular_duracion_promedio(midi_path, tolerance=0.1):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notas = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    duraciones = [n.end - n.start for n in notas]
    return np.mean(duraciones) if duraciones else 0.5



import numpy as np
bpm = 160
midi_path = "/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/OG-IT.mid"
# ================================
# Parámetros por defecto para MIDI
# ================================
DEFAULT_BPM = 160          # Tempo en BPM
DEFAULT_MEAN_VEL = 100      # Velocidad media de notas
DEFAULT_STD_VEL = 3        # Variación aleatoria de velocidad
DEFAULT_DURATION = calcular_duracion_promedio(midi_path)

def random_vel(mean_vel=DEFAULT_MEAN_VEL, std_vel=DEFAULT_STD_VEL):
    vel = int(np.random.normal(mean_vel, std_vel))
    return min(max(vel, 0), 127)  # rango MIDI válido



def arrgmt_sus1(voicing, mean_vel=DEFAULT_MEAN_VEL, std_vel=DEFAULT_STD_VEL, time=0, duration=None):
    out = []
    if duration is None:
        duration = DEFAULT_DURATION

    for p in voicing.pitches:
        vel = random_vel(mean_vel, std_vel)
        out.append({
            'type': 'note',
            'pitch': p.pitch_number,
            'time': time,
            'duration': duration,
            'vel': vel
        })
    return out


import numpy as np

def random_vel(mean_vel=DEFAULT_MEAN_VEL, std_vel=DEFAULT_STD_VEL):
    vel = int(np.random.normal(mean_vel, std_vel))
    return min(max(vel, 0), 127)  # asegurar rango MIDI

def build_midi_from_chords(generated_chords, output_path, root_note=None, bpm=160, base_pitch=48):
    new_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano = pretty_midi.Instrument(program=0)

    time = 0.0

    for chord_str in generated_chords:
        if chord_str in ['<sos>', '<eos>', '<pad>']:
            continue

        notes = [int(n) for n in chord_str.split('_') if n.isdigit()]

        # Agregar root note en octava inferior
        if notes:
            root_pitch = min(notes)
            target_pitch = root_pitch - 12

            MIN_ROOT_PITCH = 36 + 12
            while target_pitch < MIN_ROOT_PITCH:
                target_pitch += 12

            root_pitch_octave = target_pitch

            piano.notes.append(pretty_midi.Note(
                velocity=random_vel(DEFAULT_MEAN_VEL + 10, DEFAULT_STD_VEL),  # velocity aleatoria
                pitch=root_pitch_octave,
                start=time,
                end=time + max(DEFAULT_DURATION, 0.6)
            ))

        # Añadir las demás notas
        for interval in notes:
            pitch = base_pitch + interval
            pitch = max(0, min(pitch, 127))
            piano.notes.append(pretty_midi.Note(
                velocity=random_vel(),  # velocity aleatoria
                pitch=pitch,
                start=time,
                end=time + DEFAULT_DURATION
            ))

        time += DEFAULT_DURATION

    new_midi.instruments.append(piano)
    new_midi.write(output_path)


# Parámetros
# ================================

input_midi_dir = "/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO"
output_variantes_dir = "/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/variantes"
output_wav_dir = "/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/variantes_wav"
sf2_path = "/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/Steinway Grand Piano 1.2.sf2"

os.makedirs(output_variantes_dir, exist_ok=True)
os.makedirs(output_wav_dir, exist_ok=True)

# ================================
# Funciones Auxiliares (del código original)
# ================================

# (Aquí irían midi_to_chords_train, progression_to_token_seq,
# detectar_escala, generar_mejores_variantes, token_to_chord_symbol,
# build_midi_from_chords, etc. — todas las definidas arriba)


# ================================
# Cargar diccionarios y modelo
# ================================

with open("/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/token_to_id_train.pkl", "rb") as f:
    token_to_id_train = pickle.load(f)

with open("/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/id_to_token_train.pkl", "rb") as f:
    id_to_token_train = pickle.load(f)

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

model.load_state_dict(torch.load("/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/cvae_lstm_model.pth", map_location=device))
model.eval()

fs = FluidSynth(sf2_path)
fs.bpm = 160

# ================================
# Función para convertir MIDI a WAV con configuración personalizada
# ================================
import subprocess
import shlex

SF2_PATH = "/Users/pepebeats/Desktop/Semi-Supervised-CVAE-LSTM/test4EVO/Steinway Grand Piano 1.2.sf2"

def midi_to_wav_fluidsynth(midi_path, wav_path=None, sf2_path=SF2_PATH, verbose=True):
    if wav_path is None:
        # Usar como nombre la progresion con simbolos
        wav_path = os.path.splitext(midi_path)[0] + ".wav"

    if not wav_path.lower().endswith(".wav"):
        wav_path += ".wav"

    command = f"fluidsynth -F {shlex.quote(wav_path)} -T wav -g 1 -R 1 {shlex.quote(sf2_path)} {shlex.quote(midi_path)}"


    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    while process.poll() is None:
        line = process.stdout.readline()

    out, err = process.communicate()

    return wav_path

# ================================
# Procesar todos los MIDI de entrada
# ================================
archivos_midi = [f for f in os.listdir(input_midi_dir) if f.lower().endswith(".mid")]

for archivo in archivos_midi:
    try:
        midi_path = os.path.join(input_midi_dir, archivo)

        # Extraer progresión de acordes
        chords_prog = midi_to_chords_train(midi_path)
        if not chords_prog:
            print(f"[⚠️] No se detectaron acordes en {archivo}, saltando...")
            continue

        entrada_tokens = progression_to_token_seq(chords_prog, token_to_id_train)
        x_input = torch.tensor(entrada_tokens, dtype=torch.long).unsqueeze(0).to(device)

        notas_midi = [
            60 + int(p)
            for tok in entrada_tokens
            for p in id_to_token_train[tok].split('_') if p.isdigit()
        ]

        escala_mod12 = detectar_escala(notas_midi)
        root_note = min(notas_midi) % 12

        # Generar mejores variantes
        mejores_variantes = generar_mejores_variantes_con_longitud(
            model, x_input,
            token_to_id=token_to_id_train,
            id_to_token=id_to_token_train,
            escala_mod12=escala_mod12,
            root_note=root_note,
            longitud_objetivo=len(entrada_tokens),
            num_generadas=200,
            num_mejores=10
        )

        # Guardar variantes y WAVs usando Fluidsynth por línea de comandos
        for i, (variante, loss) in enumerate(mejores_variantes):
            simbolos = [token_to_chord_symbol(ch) for ch in variante]

            midi_output_path = os.path.join(output_variantes_dir, f"{os.path.splitext(archivo)[0]}_VAR_{i+1}.mid")
            build_midi_from_chords(variante, midi_output_path, root_note=root_note, base_pitch=60, bpm=bpm)

            wav_output_path = os.path.join(output_wav_dir, f"{os.path.splitext(archivo)[0]}_VAR_{i+1}.wav")


            midi_to_wav_fluidsynth(midi_output_path, wav_output_path, sf2_path=SF2_PATH)
            #imprimier simobolos de la progresion
            progresion = ' | '.join([token_to_chord_symbol(ch) for ch in variante])
            print(f"[✅] Progresion {i+1} WAV guardada: {progresion} → {wav_output_path}")
    except Exception as e:
        print(f"[❌] Error procesando {archivo}: {e}")
