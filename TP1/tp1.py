"""
TB065-86.05 - Señales y Sistemas
Trabajo Práctico Especial 1: Análisis de la señal de habla
Palabra analizada: "LAPACHOS"

Instrucciones:
  1. Grabá los audios y guardalos como WAV (mono, 44100 Hz recomendado).
  2. Ajustá los nombres de archivo y los rangos de tiempo en la sección
     "=== CONFIGURACIÓN ===" antes de correr el script.
  3. Ejecutá: python analisis_habla.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram

# =============================================================================
#  === CONFIGURACIÓN — editá estos valores según tus audios ===
# =============================================================================

AUDIO_LENTO = "lapachos_lento.wav"   # nombre del archivo de voz lenta
AUDIO_RAPIDO = "lapachos_rapido.wav"  # nombre del archivo de voz rápida

# ── Punto 1 ──────────────────────────────────────────────────────────────────
# Regiones de la señal LENTA (en segundos).
# Ajustá los valores t_inicio y t_fin mirando la forma de onda.
# "lapachos": L-A-P-A-CH-O-S

REGIONES_PERIODICAS = [
    {"nombre": "/l/ + /a/ (LA)",  "t_ini": 0.00, "t_fin": 0.316},
    {"nombre": "/a/ (PA)",        "t_ini": 0.447, "t_fin": 0.772},
    {"nombre": "/o/ (CHO)",       "t_ini": 0.906, "t_fin": 1.181},
]

REGIONES_NO_PERIODICAS = [
    {"nombre": "/p/ (explosivo)",  "t_ini": 0.316, "t_fin": 0.447},
    {"nombre": "/ch/ (africado)", "t_ini": 0.810, "t_fin": 0.906},
    {"nombre": "/s/ (fricativo)", "t_ini": 1.187, "t_fin": 1.567},
]

# ── Punto 2 ──────────────────────────────────────────────────────────────────
# Segmentos de [a] — misma cantidad en lenta y rápida
SEGS_LENTA = [
    {"nombre": "[a] de LA (lento)",  "t_ini": 0.15, "t_fin": 0.20, "tipo": "periodico"},
    {"nombre": "[a] de PA (lento)",  "t_ini": 0.5, "t_fin": 0.525, "tipo": "periodico"},
    {"nombre": "[s] (lento)",        "t_ini": 1.25, "t_fin": 1.30, "tipo": "no_periodico"},
    {"nombre": "[l] de LA (lento)",  "t_ini": 0.017, "t_fin": 0.060, "tipo": "periodico"},
    {"nombre": "[o] de CHOS (lento)","t_ini": 0.952, "t_fin": 1.062, "tipo": "periodico"},
]
 
# Lista ordenada de segmentos para la señal RÁPIDA — mismo orden que lenta.
SEGS_RAPIDA = [
    {"nombre": "[a] de LA (rápido)",  "t_ini": 0.070, "t_fin": 0.125, "tipo": "periodico"},
    {"nombre": "[a] de PA (rápido)",  "t_ini": 0.278, "t_fin": 0.353, "tipo": "periodico"},
    {"nombre": "[s] (rápido)",        "t_ini": 0.75, "t_fin": 0.80, "tipo": "no_periodico"},
    {"nombre": "[l] de LA (rápido)",  "t_ini": 0.00, "t_fin": 0.025, "tipo": "periodico"},
    {"nombre": "[o] de CHOS (rápido)","t_ini": 0.538, "t_fin": 0.604, "tipo": "periodico"},
]

 
# ── Punto 3 ──────────────────────────────────────────────────────────────────
# Vocales a analizar con FFT — ajustá los tiempos
VOCALES_LENTA = [
    {"nombre": "Vocal /a/ (LA)",  "t_ini": 0.075, "t_fin": 0.317},
    {"nombre": "Vocal /a/ (PA)",  "t_ini": 0.503, "t_fin": 0.676},
    {"nombre": "Vocal /o/ (CHO)", "t_ini": 0.938, "t_fin": 1.069},
]
 # Vocales de la señal RÁPIDA — ajustá los tiempos con visualizar_onda()
VOCALES_RAPIDA = [
    {"nombre": "Vocal /a/ (LA)",  "t_ini": 0.070, "t_fin": 0.125},
    {"nombre": "Vocal /a/ (PA)",  "t_ini": 0.278, "t_fin": 0.353},
    {"nombre": "Vocal /o/ (CHO)", "t_ini": 0.538, "t_fin": 0.604},
]
# =============================================================================
#  Utilidades
# =============================================================================
 
COLORS = {
    "periodica":    "#2196F3",   # azul
    "no_periodica": "#F44336",   # rojo
    "segmento":     "#4CAF50",   # verde
    "fft_multi":    "#9C27B0",   # violeta
    "fft_one":      "#FF9800",   # naranja
}
 
 
def cargar_audio(ruta: str):
    """Carga un archivo WAV y devuelve (fs, señal_normalizada)."""
    fs, data = wavfile.read(ruta)
    if data.ndim > 1:          # stereo → mono
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    data /= np.abs(data).max() # normalizar a ±1
    return fs, data
 
 
def tiempo_a_muestras(t_ini, t_fin, fs):
    return int(t_ini * fs), int(t_fin * fs)
 
 
def estimar_periodo_autocorr(señal, fs, f_min=60, f_max=400):
    """
    Estima el período fundamental (T0) y la frecuencia fundamental (F0)
    usando autocorrelación. Busca el primer pico entre f_min y f_max Hz.
    """
    lag_min = int(fs / f_max)
    lag_max = int(fs / f_min)
    n = len(señal)
    autocorr = np.correlate(señal, señal, mode="full")
    autocorr = autocorr[n - 1:]          # parte positiva
    autocorr /= autocorr[0]              # normalizar
 
    ventana = autocorr[lag_min:lag_max]
    if len(ventana) == 0:
        return None, None
    idx_pico = np.argmax(ventana) + lag_min
    T0 = idx_pico / fs
    F0 = 1.0 / T0
    return T0, F0
 
 
def agregar_regiones(ax, regiones, color, alpha=0.25, ymin=0, ymax=1):
    """Sombrea regiones en un eje y añade etiquetas."""
    for r in regiones:
        ax.axvspan(r["t_ini"], r["t_fin"],
                   color=color, alpha=alpha,
                   ymin=ymin, ymax=ymax)
        mid = (r["t_ini"] + r["t_fin"]) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.88,
                r["nombre"], ha="center", fontsize=7,
                color=color, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
 
 
# =============================================================================
#  PUNTO 1 — Señal lenta con regiones periódicas / no periódicas
# =============================================================================
 
def punto1(ruta):
    print("\n=== PUNTO 1 ===")
    fs, señal = cargar_audio(ruta)
    t = np.arange(len(señal)) / fs
 
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle('Punto 1 — Señal de voz lenta: "Lapachos"\nRegiones periódicas y no periódicas',
                 fontsize=13, fontweight="bold")
 
    ax.plot(t, señal, color="steelblue", lw=0.6)
    ax.set_ylabel("Amplitud (normalizada)")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
 
    agregar_regiones(ax, REGIONES_PERIODICAS,    COLORS["periodica"],    alpha=0.22)
    agregar_regiones(ax, REGIONES_NO_PERIODICAS, COLORS["no_periodica"], alpha=0.22)
 
    leyenda = [
        mpatches.Patch(color=COLORS["periodica"],    alpha=0.6, label="Periódica (vocales / sonantes)"),
        mpatches.Patch(color=COLORS["no_periodica"], alpha=0.6, label="No periódica (fricativos / explosivos)"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=8)
 
    plt.tight_layout()
    plt.savefig("punto1_señal_lenta.png", dpi=150)
    plt.show()
    print("Figura guardada: punto1_señal_lenta.png")
 
 
# =============================================================================
#  PUNTO 2 — Segmentos [a] y [s]; estimación de período y frecuencia
# =============================================================================
 
def analizar_segmento(señal, fs, t_ini, t_fin, nombre, ax, color, calcular_periodo=True):
    """Grafica el segmento de la señal y opcionalmente estima su período (solo consola)."""
    n_ini, n_fin = tiempo_a_muestras(t_ini, t_fin, fs)
    seg = señal[n_ini:n_fin]
    t_seg = np.arange(len(seg)) / fs + t_ini
 
    ax.plot(t_seg, seg, color=color, lw=0.8)
    ax.set_ylabel("Amplitud")
    ax.set_title(f"{nombre}  [{t_ini:.3f} s – {t_fin:.3f} s]", fontsize=9)
    ax.set_xlabel("Tiempo (s)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
 
    if calcular_periodo:
        T0, F0 = estimar_periodo_autocorr(seg, fs)
        if T0 and T0 < 0.05:
            print(f"  {nombre}: T₀ = {T0*1000:.2f} ms  |  F₀ = {F0:.1f} Hz")
        else:
            print(f"  {nombre}: no se encontró periodicidad clara.")
        return T0, F0
 
    return None, None
 
 
def punto2(ruta_lenta, ruta_rapida):
    """
    Grafica los segmentos en el orden definido en SEGS_LENTA / SEGS_RAPIDA.
    Columna izquierda = lenta | columna derecha = rápida.
    Los segmentos periódicos estiman T₀ y F₀; los no periódicos no.
    """
    print("\n=== PUNTO 2 ===")
 
    fs_l, señal_l = cargar_audio(ruta_lenta)
    fs_r, señal_r = cargar_audio(ruta_rapida)
 
    assert len(SEGS_LENTA) == len(SEGS_RAPIDA), \
        "SEGS_LENTA y SEGS_RAPIDA deben tener la misma cantidad de entradas."
 
    n_filas = len(SEGS_LENTA)
    fig, axes = plt.subplots(n_filas, 2, figsize=(13, 5 * n_filas))
 
    for i, (seg_l, seg_r) in enumerate(zip(SEGS_LENTA, SEGS_RAPIDA)):
        es_periodico = seg_l["tipo"] == "periodico"
        color = COLORS["periodica"] if es_periodico else COLORS["no_periodica"]
 
        analizar_segmento(señal_l, fs_l, seg_l["t_ini"], seg_l["t_fin"],
                          seg_l["nombre"], axes[i, 0], color,
                          calcular_periodo=es_periodico)
        analizar_segmento(señal_r, fs_r, seg_r["t_ini"], seg_r["t_fin"],
                          seg_r["nombre"], axes[i, 1], color,
                          calcular_periodo=es_periodico)
 
    axes[0, 0].set_title("◀ Señal LENTA\n" + axes[0, 0].get_title(), fontsize=9)
    axes[0, 1].set_title("Señal RÁPIDA ▶\n" + axes[0, 1].get_title(), fontsize=9)
 
    plt.tight_layout()
    plt.subplots_adjust(hspace=1)
    plt.savefig("punto2_segmentos.png", dpi=150)
    plt.show()
    print("Figura guardada: punto2_segmentos.png")
 
# =============================================================================
#  PUNTO 3 — FFT de las vocales (un período vs. varios períodos)
# =============================================================================
def fft_segmento(señal, fs, t_ini, t_fin):
    n_ini, n_fin = tiempo_a_muestras(t_ini, t_fin, fs)
    seg = señal[n_ini:n_fin]
    N = len(seg)
    ventana = np.hanning(N)
    Y = np.abs(fft(seg * ventana))[:N // 2]
    freqs = fftfreq(N, 1 / fs)[:N // 2]
    return freqs, Y


def un_periodo(señal, fs, t_ini, T0_s):
    """Devuelve el segmento de exactamente un período."""
    n_ini = int(t_ini * fs)
    n_fin = n_ini + int(T0_s * fs)
    seg = señal[n_ini:n_fin]
    N = len(seg)
    ventana = np.hanning(N)
    Y = np.abs(fft(seg * ventana))[:N // 2]
    freqs = fftfreq(N, 1 / fs)[:N // 2]
    return freqs, Y


def hallar_formantes(freqs, magnitudes, n_formantes=3, f_min=200, f_max=4000):
    """
    Estima los formantes buscando picos en la envolvente suavizada del espectro.
    Suavizar primero evita confundir armónicos individuales con formantes.
    """
    from scipy.signal import find_peaks, savgol_filter

    mask = (freqs >= f_min) & (freqs <= f_max)
    f_v = freqs[mask]
    m_v = magnitudes[mask]

    if len(f_v) < 10:
        return []

    # Suavizar la envolvente espectral para eliminar los picos de armónicos
    # ventana = ~200 Hz de ancho, debe ser impar
    resolucion_hz = f_v[1] - f_v[0] if len(f_v) > 1 else 1
    ventana_muestras = int(200 / resolucion_hz)
    if ventana_muestras % 2 == 0:
        ventana_muestras += 1
    ventana_muestras = max(5, ventana_muestras)

    envolvente = savgol_filter(m_v, window_length=ventana_muestras, polyorder=3)

    # Buscar picos con distancia mínima de ~300 Hz entre formantes
    dist_muestras = max(1, int(300 / resolucion_hz))
    picos, props = find_peaks(envolvente, distance=dist_muestras, prominence=envolvente.max() * 0.05)

    if len(picos) == 0:
        return []

    # Tomar los primeros n_formantes picos en orden de frecuencia (de menor a mayor)
    formantes = [(f_v[picos[i]], envolvente[picos[i]]) for i in range(min(n_formantes, len(picos)))]
    return formantes


def graficar_fft_vocales(vocales, señal, fs, titulo, nombre_archivo):
    """Genera la figura de FFT para una señal (lenta o rápida)."""
    n_vocales = len(vocales)
    fig, axes = plt.subplots(n_vocales, 2, figsize=(14, 5 * n_vocales))
    if n_vocales == 1:
        axes = np.array([axes])
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    for i, vocal in enumerate(vocales):
        nombre = vocal["nombre"]
        t_ini, t_fin = vocal["t_ini"], vocal["t_fin"]

        n0, n1 = tiempo_a_muestras(t_ini, t_fin, fs)
        T0, F0 = estimar_periodo_autocorr(señal[n0:n1], fs)
        print(f"  {nombre}: F₀ = {F0:.1f} Hz" if F0 else f"  {nombre}: F₀ no estimado")

        ax_multi = axes[i, 0]
        ax_one   = axes[i, 1]

        # — Varios períodos —
        freqs_m, Y_m = fft_segmento(señal, fs, t_ini, t_fin)
        ax_multi.plot(freqs_m, 20 * np.log10(Y_m + 1e-10),
                      color=COLORS["fft_multi"], lw=0.7)
        ax_multi.set_xlim(0, 5000)
        ax_multi.set_title(f"{nombre} — Varios períodos", fontsize=9)
        ax_multi.grid(True, alpha=0.3)

        formantes = hallar_formantes(freqs_m, Y_m)
        for fi, (f_f, m_f) in enumerate(formantes):
            m_dB = 20 * np.log10(m_f + 1e-10)
            ax_multi.axvline(f_f, color="red", lw=1, linestyle="--", alpha=0.7)
            ax_multi.text(f_f + 30, m_dB - 5, f"F{fi+1}={f_f:.0f}Hz",
                          fontsize=7, color="red")
            print(f"    F{fi+1} (multi-período) = {f_f:.1f} Hz")

        # — Un solo período —
        if T0 and T0 > 0:
            freqs_o, Y_o = un_periodo(señal, fs, t_ini + 0.01, T0)
            ax_one.plot(freqs_o, 20 * np.log10(Y_o + 1e-10),
                        color=COLORS["fft_one"], lw=0.7)
            ax_one.set_xlim(0, 5000)
            ax_one.set_title(f"{nombre} — Un período (T₀≈{T0*1000:.1f} ms)", fontsize=9)

            formantes_o = hallar_formantes(freqs_o, Y_o)
            for fi, (f_f, m_f) in enumerate(formantes_o):
                m_dB = 20 * np.log10(m_f + 1e-10)
                ax_one.axvline(f_f, color="darkred", lw=1, linestyle="--", alpha=0.7)
                ax_one.text(f_f + 30, m_dB - 5, f"F{fi+1}={f_f:.0f}Hz",
                            fontsize=7, color="darkred")
                print(f"    F{fi+1} (1 período)    = {f_f:.1f} Hz")
        else:
            ax_one.text(0.5, 0.5, "T₀ no estimado\n(ajustar tiempos)",
                        ha="center", va="center", transform=ax_one.transAxes, fontsize=10)
            ax_one.set_title(f"{nombre} — Un período (no disponible)", fontsize=9)

        for ax in [ax_multi, ax_one]:
            ax.set_ylabel("Magnitud (dB)")
            ax.set_xlabel("Frecuencia (Hz)")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, top=0.87)
    plt.savefig(nombre_archivo, dpi=150)
    plt.show()
    print(f"Figura guardada: {nombre_archivo}")


def punto3(ruta_lenta, ruta_rapida):
    print("\n=== PUNTO 3 ===")
    fs_l, señal_l = cargar_audio(ruta_lenta)
    fs_r, señal_r = cargar_audio(ruta_rapida)

    print("\n— Señal LENTA —")
    graficar_fft_vocales(
        VOCALES_LENTA, señal_l, fs_l,
        titulo='Punto 3 — FFT vocales · Señal LENTA\n(varios períodos vs. un período)',
        nombre_archivo="punto3_fft_lenta.png"
    )

    print("\n— Señal RÁPIDA —")
    graficar_fft_vocales(
        VOCALES_RAPIDA, señal_r, fs_r,
        titulo='Punto 3 — FFT vocales · Señal RÁPIDA\n(varios períodos vs. un período)',
        nombre_archivo="punto3_fft_rapida.png"
    )


 
# =============================================================================
#  AUXILIAR — Visualizador interactivo para calibrar los tiempos
# =============================================================================
 
def visualizar_onda(ruta, titulo="Señal de voz"):
    """
    Muestra la forma de onda del archivo para que puedas leer los tiempos
    y ajustar las variables de CONFIGURACIÓN arriba.
    Usá esta función primero antes de correr los puntos.
    """
    fs, señal = cargar_audio(ruta)
    t = np.arange(len(señal)) / fs
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    fig.suptitle(f"Calibración de tiempos — {ruta}", fontsize=11)
 
    ax1.plot(t, señal, lw=0.5, color="steelblue")
    ax1.set_ylabel("Amplitud")
    ax1.set_title(titulo)
    ax1.grid(True, alpha=0.3)
 
    f_sg, t_sg, Sxx = spectrogram(señal, fs=fs, nperseg=256, noverlap=200)
    ax2.pcolormesh(t_sg, f_sg, 10 * np.log10(Sxx + 1e-10),
                   shading="gouraud", cmap="inferno")
    ax2.set_ylim(0, 6000)
    ax2.set_ylabel("Frecuencia (Hz)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_title("Espectrograma")
 
    plt.tight_layout()
    plt.show()
 
 
# =============================================================================
#  MAIN
# =============================================================================
 
if __name__ == "__main__":
    import sys
 
    print("=" * 60)
    print("  TP Especial 1 — Análisis de la señal de habla")
    print("  Palabra: 'Lapachos'")
    print("=" * 60)
 
    # PASO 0: Corré esto primero para ver la forma de onda y ajustar tiempos.
    # visualizar_onda(AUDIO_LENTO, "Voz lenta — Lapachos")
    # visualizar_onda(AUDIO_RAPIDO, "Voz rápida — Lapachos")
 
    # Descomentá las líneas que necesites:
    #punto1(AUDIO_LENTO)
    #punto2(AUDIO_LENTO, AUDIO_RAPIDO)
    punto3(AUDIO_LENTO, AUDIO_RAPIDO)
