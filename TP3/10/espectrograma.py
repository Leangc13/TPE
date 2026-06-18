import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Parámetros según el tipo de espectrograma
PARAMS = {
    "angosta": {
        "nperseg_ms": 30,
        "overlap_pct": 0.75,
        "nfft_factor": 4,          # NFFT = nfft_factor × nperseg  →  zero-padding
        "titulo": "Espectrograma de Banda Angosta",
        "freq_max": 4000,
    },
    "ancha": {
        "nperseg_ms": 5,
        "overlap_pct": 0.9,
        "nfft_factor": 8,          # más factor en banda ancha porque nperseg es pequeño
        "titulo": "Espectrograma de Banda Ancha",
        "freq_max": 4000,
    },
    "ancha decimada": {
        "nperseg_ms": 2.5,
        "overlap_pct": 0.875,
        "nfft_factor": 8,          # más factor en banda ancha porque nperseg es pequeño
        "titulo": "Espectrograma de Banda Ancha",
        "freq_max": 4000,
    },

    "ancha interpolada": {
        "nperseg_ms": 10,          # El doble que la "ancha" original (5 ms * 2)
        "overlap_pct": 0.75,       # Mantenemos el mismo solapamiento
        "nfft_factor": 8,          # Mantenemos el factor de zero-padding para suavidad
        "titulo": "Espectrograma de Banda Ancha (Interpolada)",
        "freq_max": 4000,          # Mantenemos 4000 Hz para comparar visualmente
    },
}

def plot_espectrograma(archivo, tipo, nombre, rango_t=None):
    """
    archivo  : path al .wav
    tipo     : "angosta" o "ancha"
    nombre   : string descriptivo de lo que se está analizando (ej: "Lapachos lento - vocal [a]")
    rango_t  : None            → grafica todo el audio
               [t_ini, t_fin]  → grafica solo ese intervalo (en segundos)
    """
    if tipo not in PARAMS:
        raise ValueError(f"TIPO_ESPECTROGRAMA debe ser 'angosta' o 'ancha', no '{tipo}'")

    p = PARAMS[tipo]

    # Leer audio
    fs, data = wavfile.read(archivo)

    # Si es estéreo, tomar solo un canal
    if data.ndim > 1:
        data = data[:, 0]

    # Normalizar a float [-1, 1]
    data = data.astype(np.float64)
    data /= np.max(np.abs(data))

    # Recortar señal si se especifica rango_t
    duracion_total = len(data) / fs
    if rango_t is not None:
        t_ini, t_fin = rango_t
        if t_ini < 0 or t_fin > duracion_total or t_ini >= t_fin:
            raise ValueError(f"rango_t={rango_t} inválido. El audio dura {duracion_total:.3f} s")
        muestra_ini = int(t_ini * fs)
        muestra_fin = int(t_fin * fs)
        data = data[muestra_ini:muestra_fin]
    else:
        t_ini = 0

    # Calcular parámetros de la ventana
    nperseg = int(p["nperseg_ms"] * fs / 1000)
    noverlap = int(nperseg * p["overlap_pct"])
    nfft    = nperseg * p["nfft_factor"]   # zero-padding: NFFT > nperseg → mayor resolución en frecuencia

    # Calcular espectrograma
    f, t, Sxx = spectrogram(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="spectrum",
    )

    # Desplazar eje temporal al valor real dentro del audio original
    t = t + t_ini

    # Pasar a dB (escala logarítmica)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    # Filtrar hasta freq_max
    freq_mask = f <= p["freq_max"]
    f_plot = f[freq_mask]
    Sxx_plot = Sxx_dB[freq_mask, :]

    # Graficar
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{p['titulo']} – {nombre}", fontsize=13, fontweight="bold")

    # --- Espectrograma ---
    im = ax.pcolormesh(
        t, f_plot, Sxx_plot,
        shading="gouraud",
        cmap="inferno",
        vmin=Sxx_plot.max() - 60,
        vmax=Sxx_plot.max(),
    )
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Frecuencia [Hz]")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energía [dB]")

    plt.tight_layout()
    plt.savefig(f"espectrograma_{tipo}_"+ nombre +".png", dpi=150, bbox_inches="tight")
    # plt.savefig(f"espectrograma_{tipo}_"+ nombre + "[lento]" +".png", dpi=150, bbox_inches="tight")
    # plt.savefig(f"espectrograma_{tipo}_"+ nombre + "[rapido]" +".png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OK] Guardado: espectrograma_{tipo}_"+ nombre +".png")
    print(f"     fs = {fs} Hz  |  nperseg = {nperseg} muestras "
          f"({p['nperseg_ms']} ms)  |  noverlap = {noverlap}  |  nfft = {nfft} (×{p['nfft_factor']})")


VOCALES_LENTA = [
    {"nombre": "Vocal -a- (LA)",  "t_ini": 0.075, "t_fin": 0.317},
    {"nombre": "Vocal -a- (PA)",  "t_ini": 0.503, "t_fin": 0.676},
    {"nombre": "Vocal -o- (CHOS)", "t_ini": 0.940, "t_fin": 1.053},
]
 # Vocales de la señal RÁPIDA — ajustá los tiempos con visualizar_onda()
VOCALES_RAPIDA = [
    {"nombre": "Vocal -a- (LA)",  "t_ini": 0.070, "t_fin": 0.125},
    {"nombre": "Vocal -a- (PA)",  "t_ini": 0.278, "t_fin": 0.353},
    {"nombre": "Vocal -o- (CHOS)", "t_ini": 0.538, "t_fin": 0.604},
]

if __name__ == "__main__":
    #plot_espectrograma("lapachos_lento_decimada.wav",  "ancha decimada",   "LAPACHOS decimado (lento)")

    plot_espectrograma("lapachos_rapida_a_lenta_tfct_manual.wav", "ancha",   "LAPACHOS phase vocoder(rapida a lenta)")

    # Ejemplo con segmento y nombre descriptivo
    # plot_espectrograma("lapachos_lento.wav", "angosta", "Vocal [a]", rango_t=[0.5, 1.2])
    
