"""
Decimación por factor 2 de lapachos_lento.wav con filtro antialiasing
por el método de ventaneo (FIR). Devuelve solo el audio decimado.
"""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

# ─── Rutas ───────────────────────────────────────────────────────────────────
PATH_IN  = 'lapachos_rapido.wav'
PATH_OUT = 'lapachos_rapido_decimada.wav'

# ─── Parámetros ──────────────────────────────────────────────────────────────
M       = 2        # Factor de decimación
N_fir   = 101      # Orden del filtro FIR (impar → fase lineal)
VENTANA = 'hamming'

# ─── 1. Carga ────────────────────────────────────────────────────────────────
fs, x = wav.read(PATH_IN)
x = x.astype(np.float64) / 32768.0

print(f"fs = {fs} Hz")
print(f"Duración original: {len(x)/fs:.3f} s  ({len(x)} muestras)")

# ─── 2. Filtro antialiasing FIR ──────────────────────────────────────────────
Wn = 1.0 / M   # fc normalizada = fs/4
h  = signal.firwin(N_fir, Wn, window=VENTANA)

# ─── 3. Filtrar + decimar ────────────────────────────────────────────────────
x_filtrada = signal.lfilter(h, 1.0, x)

retardo    = (N_fir - 1) // 2
x_filtrada = x_filtrada[retardo:]   # compensar retardo de grupo

x_decimada = x_filtrada[::M]        # tomar 1 de cada M muestras

print(f"Duración decimada : {len(x_decimada)/fs:.3f} s  ({len(x_decimada)} muestras)")

# ─── 4. Guardar ──────────────────────────────────────────────────────────────
wav.write(PATH_OUT, fs, (x_decimada * 32767).astype(np.int16))
print(f"Archivo guardado: {PATH_OUT}")
