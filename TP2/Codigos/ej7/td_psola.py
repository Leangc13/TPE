import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

# ── Configuración ─────────────────────────────────────────────
AUDIO_FILE   = "lapachos_lento.wav"   # cambiar según archivo
RATIO        = 1.4                    # factor de pitch (hombre → mujer)
F0_MIN, F0_MAX = 80, 170              # rango de pitch del hablante (Hz)

SEGS_LENTA = [
    {"nombre": "[a] de LA (lento)",   "t_ini": 0.15,  "t_fin": 0.20},
    {"nombre": "[a] de PA (lento)",   "t_ini": 0.50,  "t_fin": 0.525},
    {"nombre": "[l] de LA (lento)",   "t_ini": 0.017, "t_fin": 0.060},
    {"nombre": "[o] de CHOS (lento)", "t_ini": 0.952, "t_fin": 1.062},
]

# ── Carga de audio ─────────────────────────────────────────────
x, fs = sf.read(AUDIO_FILE)
if x.ndim > 1:
    x = x[:, 0]           # mono
y = x.copy().astype(float)

# ── Ventana de análisis/síntesis ───────────────────────────────
def hann_window(N):
    return np.hanning(N)

# ── TD-PSOLA en un segmento ────────────────────────────────────
def td_psola_segment(sig, fs, ratio, f0_min=80, f0_max=170):
    """
    Aplica TD-PSOLA a 'sig' escalando el pitch por 'ratio'.
    Devuelve un array de la misma longitud.
    """
    N = len(sig)
    T0_min = int(fs / f0_max)   # período mínimo en muestras
    T0_max = int(fs / f0_min)   # período máximo en muestras
    win_len = 2 * T0_max        # ventana = 2 períodos máximos

    # 1. Detectar picos (marcas de pitch, GCIs aproximados)
    peaks, _ = find_peaks(sig, distance=T0_min, height=0.1 * np.max(np.abs(sig)))
    if len(peaks) < 2:
        return sig  # segmento demasiado corto o no sonoro

    # 2. Estimar período promedio entre picos consecutivos
    T0_est = int(np.median(np.diff(peaks)))
    T0_new = T0_est / ratio     # nuevo período en muestras (float)

    # 3. Síntesis: colocar granos centrados en nuevas posiciones
    out = np.zeros(N)
    w2  = win_len // 2

    t_syn = float(peaks[0])     # primera marca de síntesis
    while t_syn < N:
        # buscar pico de análisis más cercano
        dists = np.abs(peaks - t_syn)
        p_ana = peaks[np.argmin(dists)]

        # extraer grano con ventana Hann
        i0 = p_ana - w2
        i1 = p_ana + w2
        pad_l = max(0, -i0);  i0 = max(0, i0)
        pad_r = max(0, i1 - N); i1 = min(N, i1)
        grain = sig[i0:i1]
        grain = np.pad(grain, (pad_l, pad_r))[:win_len]
        grain *= hann_window(win_len)

        # superponer en posición de síntesis
        s0 = int(round(t_syn)) - w2
        s1 = s0 + win_len
        o0 = max(0, -s0);  s0 = max(0, s0)
        o1 = max(0, s1 - N); s1 = min(N, s1)
        out[s0:s1] += grain[o0: win_len - o1]

        t_syn += T0_new

    # normalizar amplitud para evitar clipping
    mx = np.max(np.abs(out))
    if mx > 0:
        out *= np.max(np.abs(sig)) / mx
    return out

# ── Procesar cada segmento sonoro ─────────────────────────────
for seg in SEGS_LENTA:
    n_ini = int(seg["t_ini"] * fs)
    n_fin = int(seg["t_fin"] * fs)
    fragmento = x[n_ini:n_fin].copy()

    print(f"Procesando: {seg['nombre']}  "
          f"[{seg['t_ini']:.3f}s – {seg['t_fin']:.3f}s]  "
          f"({n_fin - n_ini} muestras)")

    modificado = td_psola_segment(fragmento, fs, RATIO, F0_MIN, F0_MAX)
    y[n_ini:n_fin] = modificado

# ── Guardar resultado ──────────────────────────────────────────
out_file = AUDIO_FILE.replace(".wav", f"_pitch_x{RATIO}.wav")
sf.write(out_file, y, fs)
print(f"\nArchivo guardado: {out_file}")
