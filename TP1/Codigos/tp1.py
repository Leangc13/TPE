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
    # {"nombre": "Vocal /o/ (CHO)", "t_ini": 0.938, "t_fin": 1.069},
    {"nombre": "Vocal /o/ (CHO)", "t_ini": 0.940, "t_fin": 1.053},
]
 # Vocales de la señal RÁPIDA — ajustá los tiempos con visualizar_onda()
VOCALES_RAPIDA = [
    {"nombre": "Vocal /a/ (LA)",  "t_ini": 0.070, "t_fin": 0.125},
    {"nombre": "Vocal /a/ (PA)",  "t_ini": 0.278, "t_fin": 0.353},
    {"nombre": "Vocal /o/ (CHO)", "t_ini": 0.538, "t_fin": 0.604},
    # {"nombre": "Vocal /o/ (CHO)", "t_ini": 0.541, "t_fin": 0.577},
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

def estimar_f0_robusto(señal, fs, f_min=60, f_max=400):
    """
    Estima F0 como la mediana sobre ventanas de 30 ms solapadas.
    Más robusto que usar la señal completa de una vez, porque promedia
    pequeñas variaciones de tono a lo largo de la vocal.
    """
    win = int(0.03 * fs)
    hop = win // 2
    f0s = []
    for s in range(0, len(señal) - win, hop):
        chunk = señal[s : s + win]
        corr = np.correlate(chunk, chunk, mode="full")[len(chunk) - 1 :]
        corr /= corr[0]
        mn, mx = int(fs / f_max), int(fs / f_min)
        if mx >= len(corr):
            continue
        lag = np.argmax(corr[mn:mx]) + mn
        f0s.append(fs / lag)
    return float(np.median(f0s)) if f0s else 150.0


def fft_segmento(señal, fs, t_ini, t_fin):
    """FFT con ventana de Hanning sobre el segmento completo (varios períodos)."""
    n_ini, n_fin = tiempo_a_muestras(t_ini, t_fin, fs)
    seg = señal[n_ini:n_fin]
    N = len(seg)
    Y = np.abs(fft(seg * np.hanning(N)))[:N // 2]
    freqs = fftfreq(N, 1 / fs)[:N // 2]
    return freqs, Y


def fft_un_periodo(señal, fs, t_ini, f0_hz):
    """FFT de exactamente un período, centrado en t_ini."""
    n_ini = int(t_ini * fs)
    n_per = int(fs / f0_hz)
    seg = señal[n_ini : n_ini + n_per]
    N = len(seg)
    Y = np.abs(fft(seg * np.hanning(N)))[:N // 2]
    freqs = fftfreq(N, 1 / fs)[:N // 2]
    return freqs, Y


def hallar_formantes(freqs, magnitudes, f0_hz, n_formantes=3, f_min=300, f_max=4000):
    """
    Estima los formantes buscando picos en la envolvente suavizada del espectro.

    La ventana de suavizado Savitzky-Golay se adapta al caso:
    - Varios períodos (muchos puntos): ventana ≈ 1.5×F0 en Hz → aplana armónicos
      pero preserva separación entre formantes.
    - Un solo período (pocos puntos, Δf = F0): el espectro ya es muy grueso;
      se usa ventana mayor (≈ 2.5×F0) para suprimir el rizado residual.
    Una ventana fija en Hz falla con F0 bajas (como la /o/ a ~107 Hz).
    """
    from scipy.signal import find_peaks, savgol_filter

    mask = (freqs >= f_min) & (freqs <= f_max)
    f_v, m_v = freqs[mask], magnitudes[mask]
    if len(f_v) < 10:
        return [], None, None

    resol = f_v[1] - f_v[0]

    # Detectar si estamos en modo "un período" (resolución ≈ F0)
    un_periodo = resol > f0_hz * 0.7

    if un_periodo:
        # Con un solo período hay muy pocos puntos: suavizado más agresivo
        win = max(5, int(2.5 * f0_hz / resol))
    else:
        win = max(5, int(1.5 * f0_hz / resol))

    if win % 2 == 0:
        win += 1
    win = min(win, len(f_v) - 2)

    env = savgol_filter(m_v, window_length=win, polyorder=3)
    env = np.clip(env, 0, None)

    dist = max(1, int(300 / resol))
    # 0.10 del máximo: descarta picos espurios de baja energía (ej. armónicos
    # residuales o lóbulos laterales de la envolvente) sin perder formantes reales
    picos, _ = find_peaks(env, distance=dist, prominence=env.max() * 0.10)

    formantes = [(f_v[p], env[p]) for p in picos[:n_formantes]]
    return formantes, f_v, env


def _graficar_un_panel(ax, freqs, Y, f0_hz, nombre, color_espectro, color_env,
                       color_fmt, mostrar_formantes=True):
    """
    Dibuja espectro normalizado (0–1, escala lineal) + envolvente suavizada
    + marcas de formantes en un eje dado.

    Se usa escala lineal normalizada en vez de dB porque los valles entre
    armónicos bajan a -200 dB y comprimen todo el contenido útil en una
    franja muy pequeña del eje Y.

    Retorna lista de formantes: [(freq_hz, mag_norm), ...]
    """
    # ── Espectro normalizado (lineal) ──────────────────────────────────
    Y_norm = Y / (Y.max() + 1e-10)
    mask_plot = freqs <= 4500
    ax.plot(freqs[mask_plot], Y_norm[mask_plot],
            color=color_espectro, lw=0.8, alpha=0.55, label="Espectro")

    # ── Calcular formantes (para retorno) ────────────────────────────
    formantes, f_env, env_vals = hallar_formantes(freqs, Y, f0_hz)

    # ── Ejes ───────────────────────────────────────────────────────────
    ax.set_xlim(0, 4500)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=8)
    ax.set_ylabel("Magnitud normalizada", fontsize=8)
    ax.set_title(nombre, fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    return formantes


def punto3(ruta_lenta, ruta_rapida):
    """
    Punto 3: FFT de las vocales de la señal lenta y rápida.

    Figura 1 — Comparación lenta vs rápida (varios períodos):
        3 filas (una por vocal) × 2 columnas (lenta | rápida).
        Permite comparar directamente los formantes entre velocidades.

    Figura 2 — Varios períodos vs. un período (señal lenta):
        3 filas × 2 columnas (varios períodos | un período).
        Ilustra el trade-off de resolución frecuencial.

    En consola se imprime una tabla resumen con F0 y formantes.
    """
    from scipy.signal import savgol_filter

    print("\n=== PUNTO 3 ===")
    fs_l, señal_l = cargar_audio(ruta_lenta)
    fs_r, señal_r = cargar_audio(ruta_rapida)

    # ------------------------------------------------------------------
    # Pre-calcular F0 de cada vocal (robusto, mediana sobre ventanas)
    # ------------------------------------------------------------------
    f0_lenta, f0_rapida = [], []
    for vl, vr in zip(VOCALES_LENTA, VOCALES_RAPIDA):
        n0, n1 = tiempo_a_muestras(vl["t_ini"], vl["t_fin"], fs_l)
        f0_lenta.append(estimar_f0_robusto(señal_l[n0:n1], fs_l))
        n0, n1 = tiempo_a_muestras(vr["t_ini"], vr["t_fin"], fs_r)
        f0_rapida.append(estimar_f0_robusto(señal_r[n0:n1], fs_r))

    # ------------------------------------------------------------------
    # FIGURA 1 — Lenta vs Rápida (varios períodos), una fila por vocal
    # ------------------------------------------------------------------
    n_v = len(VOCALES_LENTA)
    fig1, axes1 = plt.subplots(n_v, 2, figsize=(14, 5 * n_v))
    fig1.suptitle(
        "Punto 3 — Espectro FFT: señal LENTA vs RÁPIDA\n(segmento completo de cada vocal)",
        fontsize=12, fontweight="bold",
    )

    print("\n{'Vocal':<20} {'Señal':<8} {'F0 (Hz)':>8}  {'F1':>8}  {'F2':>8}  {'F3':>8}")
    print("-" * 70)

    resumen = []   # guardar para figura de tabla (opcional)

    for i, (vl, vr) in enumerate(zip(VOCALES_LENTA, VOCALES_RAPIDA)):
        nombre_vocal = vl["nombre"].replace("Vocal ", "")

        # — Lenta —
        freqs_l, Y_l = fft_segmento(señal_l, fs_l, vl["t_ini"], vl["t_fin"])
        fmts_l = _graficar_un_panel(
            axes1[i, 0], freqs_l, Y_l, f0_lenta[i],
            nombre=f"{nombre_vocal} — LENTA  (F₀≈{f0_lenta[i]:.0f} Hz)",
            color_espectro=COLORS["fft_multi"],
            color_env="#1565C0",
            color_fmt="#0D47A1",
        )
        # — Rápida —
        freqs_r, Y_r = fft_segmento(señal_r, fs_r, vr["t_ini"], vr["t_fin"])
        fmts_r = _graficar_un_panel(
            axes1[i, 1], freqs_r, Y_r, f0_rapida[i],
            nombre=f"{nombre_vocal} — RÁPIDA  (F₀≈{f0_rapida[i]:.0f} Hz)",
            color_espectro=COLORS["fft_one"],
            color_env="#B71C1C",
            color_fmt="#7F0000",
        )


        # Imprimir resumen en consola
        def _fmt_row(señal_str, f0, fmts):
            cols = [f"{f:.0f}" for f, _ in fmts]
            cols += ["—"] * (3 - len(cols))
            print(f"  {nombre_vocal:<18} {señal_str:<8} {f0:>8.1f}  "
                  f"{cols[0]:>8}  {cols[1]:>8}  {cols[2]:>8}")

        _fmt_row("lenta",  f0_lenta[i],  fmts_l)
        _fmt_row("rápida", f0_rapida[i], fmts_r)
        resumen.append((nombre_vocal, f0_lenta[i], f0_rapida[i], fmts_l, fmts_r))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, top=0.90)
    plt.savefig("punto3_fft_lenta_vs_rapida.png", dpi=150)
    plt.show()
    print("\nFigura guardada: punto3_fft_lenta_vs_rapida.png")

    # ------------------------------------------------------------------
    # FIGURA 2 — Varios períodos vs. un período (señal LENTA)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(n_v, 2, figsize=(14, 5 * n_v))
    fig2.suptitle(
        "Punto 3 — Varios períodos vs. un período · Señal LENTA\n"
        "(comparación de resolución frecuencial)",
        fontsize=12, fontweight="bold",
    )

    for i, vl in enumerate(VOCALES_LENTA):
        nombre_vocal = vl["nombre"].replace("Vocal ", "")
        f0 = f0_lenta[i]
        T0 = 1.0 / f0

        # Varios períodos
        freqs_m, Y_m = fft_segmento(señal_l, fs_l, vl["t_ini"], vl["t_fin"])
        dur_ms = (vl["t_fin"] - vl["t_ini"]) * 1000
        _graficar_un_panel(
            axes2[i, 0], freqs_m, Y_m, f0,
            nombre=f"{nombre_vocal} — Varios períodos ({dur_ms:.0f} ms, Δf={freqs_m[1]:.1f} Hz)",
            color_espectro=COLORS["fft_multi"],
            color_env="#1565C0",
            color_fmt="#0D47A1",
        )

        # Un solo período
        freqs_o, Y_o = fft_un_periodo(señal_l, fs_l, vl["t_ini"] + 0.01, f0)
        fmts_1p = _graficar_un_panel(
            axes2[i, 1], freqs_o, Y_o, f0,
            nombre=f"{nombre_vocal} — Un período (T₀={T0*1000:.2f} ms, Δf={f0:.0f} Hz)",
            color_espectro=COLORS["fft_one"],
            color_env="#E65100",
            color_fmt="#BF360C",
            mostrar_formantes=True,
        )
        # Imprimir comparación en consola
        fmts_multi_str = [f"F{k+1}={fv:.0f}" for k, (fv, _) in enumerate(
            hallar_formantes(freqs_m, Y_m, f0)[0])]
        fmts_1p_str   = [f"F{k+1}={fv:.0f}" for k, (fv, _) in enumerate(fmts_1p)]
        print(f"    {nombre_vocal} | varios per.: {fmts_multi_str}  →  1 período: {fmts_1p_str}  (Δf={f0:.0f} Hz)")
        # Anotación de resolución
        axes2[i, 1].text(
            0.98, 0.03,
            f"Resolución frecuencial = F₀ = {f0:.0f} Hz\n"
            f"→ picos aproximados al múltiplo de F₀ más cercano",
            transform=axes2[i, 1].transAxes,
            ha="right", va="bottom", fontsize=7.5,
            bbox=dict(facecolor="lightyellow", alpha=0.85, edgecolor="gray", pad=3),
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, top=0.90)
    plt.savefig("punto3_fft_periodos.png", dpi=150)
    plt.show()
    print("Figura guardada: punto3_fft_periodos.png")

    # ------------------------------------------------------------------
    # FIGURA 3 — Varios períodos vs. un período (señal RÁPIDA)
    # ------------------------------------------------------------------
    fig3, axes3 = plt.subplots(n_v, 2, figsize=(14, 5 * n_v))
    fig3.suptitle(
        "Punto 3 — Varios períodos vs. un período · Señal RÁPIDA\n"
        "(comparación de resolución frecuencial)",
        fontsize=12, fontweight="bold",
    )

    for i, vr in enumerate(VOCALES_RAPIDA):
        nombre_vocal = vr["nombre"].replace("Vocal ", "")
        f0 = f0_rapida[i]
        T0 = 1.0 / f0

        # Varios períodos
        freqs_m, Y_m = fft_segmento(señal_r, fs_r, vr["t_ini"], vr["t_fin"])
        dur_ms = (vr["t_fin"] - vr["t_ini"]) * 1000
        _graficar_un_panel(
            axes3[i, 0], freqs_m, Y_m, f0,
            nombre=f"{nombre_vocal} — Varios períodos ({dur_ms:.0f} ms, Δf={freqs_m[1]:.1f} Hz)",
            color_espectro=COLORS["fft_multi"],
            color_env="#1565C0",
            color_fmt="#0D47A1",
        )

        # Un solo período
        freqs_o, Y_o = fft_un_periodo(señal_r, fs_r, vr["t_ini"] + 0.005, f0)
        fmts_1p = _graficar_un_panel(
            axes3[i, 1], freqs_o, Y_o, f0,
            nombre=f"{nombre_vocal} — Un período (T₀={T0*1000:.2f} ms, Δf={f0:.0f} Hz)",
            color_espectro=COLORS["fft_one"],
            color_env="#E65100",
            color_fmt="#BF360C",
            mostrar_formantes=True,
        )
        # Imprimir comparación en consola
        fmts_multi_str = [f"F{k+1}={fv:.0f}" for k, (fv, _) in enumerate(
            hallar_formantes(freqs_m, Y_m, f0)[0])]
        fmts_1p_str   = [f"F{k+1}={fv:.0f}" for k, (fv, _) in enumerate(fmts_1p)]
        print(f"    {nombre_vocal} (rápida) | varios per.: {fmts_multi_str}  →  1 período: {fmts_1p_str}  (Δf={f0:.0f} Hz)")
        # Anotación de resolución
        axes3[i, 1].text(
            0.98, 0.03,
            f"Resolución frecuencial = F₀ = {f0:.0f} Hz\n"
            f"→ picos aproximados al múltiplo de F₀ más cercano",
            transform=axes3[i, 1].transAxes,
            ha="right", va="bottom", fontsize=7.5,
            bbox=dict(facecolor="lightyellow", alpha=0.85, edgecolor="gray", pad=3),
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, top=0.90)
    plt.savefig("punto3_fft_periodos_rapida.png", dpi=150)
    plt.show()
    print("Figura guardada: punto3_fft_periodos_rapida.png")




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