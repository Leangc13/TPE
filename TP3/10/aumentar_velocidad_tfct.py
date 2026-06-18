import librosa
import soundfile as sf
import numpy as np

def aumentar_velocidad_tfct(archivo_entrada, archivo_salida):
    """
    Carga un audio, calcula su TFCT, elimina las columnas impares de la matriz
    espectral (reduciendo el tiempo a la mitad) y recupera el audio con la iTFCT.
    """
    print(f"Leyendo: {archivo_entrada}...")
    y_original, sr = librosa.load(archivo_entrada, sr=None)
    
    # Parámetros estándar para voz
    n_fft = 2048
    hop_length = 512
    
    # ==========================================
    # PASO 1: Transformada de Fourier de Corto Tiempo
    # ==========================================
    print("1. Calculando la TFCT...")
    D_original = librosa.stft(y_original, n_fft=n_fft, hop_length=hop_length)
    
    # ==========================================
    # PASO 2: Modificación de la Matriz (Eliminar columnas)
    # ==========================================
    print("2. Eliminando 1 de cada 2 columnas de la TFCT...")
    # Usamos slicing de NumPy [filas, columnas].
    # ':' selecciona todas las frecuencias (filas).
    # '::2' selecciona las columnas saltando de a 2 (0, 2, 4, 6...).
    D_modificada = D_original[:, ::2]
    
    # ==========================================
    # PASO 3: Síntesis Temporal (iTFCT)
    # ==========================================
    print("3. Recuperando la señal temporal con la iTFCT...")
    y_transformada = librosa.istft(D_modificada, hop_length=hop_length)
    
    # Guardar el nuevo archivo de audio
    sf.write(archivo_salida, y_transformada, sr)
    print(f"¡Éxito! Nuevo audio guardado en: {archivo_salida}")

# ==========================================
# EJECUCIÓN
# ==========================================
archivo_in = 'lapachos_lento.wav'
archivo_out = 'lapachos_lenta_a_rapida_tfct.wav'

aumentar_velocidad_tfct(archivo_in, archivo_out)
