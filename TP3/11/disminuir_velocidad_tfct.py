import librosa
import soundfile as sf
import numpy as np

def disminuir_velocidad_tfct(archivo_entrada, archivo_salida):
    print(f"Leyendo: {archivo_entrada}...")
    y_original, sr = librosa.load(archivo_entrada, sr=None)
    
    n_fft = 2048
    hop_length = 512
    
    # ==========================================
    # 1. TFCT: Se aplica la TFCT a la señal
    # ==========================================
    print("1. Calculando la TFCT...")
    D_original = librosa.stft(y_original, n_fft=n_fft, hop_length=hop_length)
    
    # ==========================================
    # 2. Interpolación: Agregando columnas (Duplicando)
    # ==========================================
    print("2. Interpolando la matriz (duplicando columnas)...")
    # np.repeat repite cada elemento a lo largo del eje especificado.
    # axis=1 significa que repetirá las columnas. Cada columna aparecerá 2 veces seguidas.
    D_modificada = np.repeat(D_original, repeats=2, axis=1)
    
    # ==========================================
    # 3. Síntesis Temporal: iTFCT
    # ==========================================
    print("3. Recuperando la señal temporal con la iTFCT...")
    y_transformada = librosa.istft(D_modificada, hop_length=hop_length)
    
    sf.write(archivo_salida, y_transformada, sr)
    print(f"¡Éxito! Nuevo audio guardado en: {archivo_salida}")

# ==========================================
# EJECUCIÓN
# ==========================================
archivo_in = 'lapachos_rapido.wav'
archivo_out = 'lapachos_lenta_tfct_manual.wav'

disminuir_velocidad_tfct(archivo_in, archivo_out)
