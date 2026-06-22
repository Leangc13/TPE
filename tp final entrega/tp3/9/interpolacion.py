import numpy as np
import librosa
import soundfile as sf
from scipy.signal import firwin, lfilter

def interpolacion_con_filtro(archivo_entrada, archivo_salida):
    print(f"Leyendo: {archivo_entrada}...")
    y_original, sr = librosa.load(archivo_entrada, sr=None)
    N = len(y_original)
    
    # ==========================================
    # PASO 1: Expansión (Zero-stuffing) L = 2
    # ==========================================
    print("1. Insertando ceros (Zero-stuffing)...")
    # Creamos un arreglo del doble de tamaño lleno de ceros
    y_expandida = np.zeros(2 * N)
    # Colocamos las muestras originales en las posiciones pares
    y_expandida[0::2] = y_original
    
    # ==========================================
    # PASO 2: Diseño del Filtro Pasabajos
    # ==========================================
    print("2. Diseñando y aplicando filtro pasabajos...")
    # Diseñamos un filtro FIR pasabajos (ventana de Hamming por defecto)
    # Frecuencia de corte normalizada a la mitad de Nyquist: 1/L = 0.5
    numtaps = 101 # Orden del filtro (cantidad de coeficientes)
    corte = 0.5 
    h = firwin(numtaps, corte)
    
    # Multiplicamos los coeficientes por L (2) para compensar la pérdida 
    # de energía (amplitud) al insertar ceros
    h = h * 2.0 
    
    # ==========================================
    # PASO 3: Filtrado (Convolución)
    # ==========================================
    # Pasamos la señal expandida por el filtro pasabajos
    y_interpolada = lfilter(h, 1.0, y_expandida)
    
    print("Guardando el nuevo archivo de audio...")
    sf.write(archivo_salida, y_interpolada, sr)
    print(f"¡Éxito! Nuevo audio guardado en: {archivo_salida}")

# Ejecución
archivo_de_entrada = 'lapachos_lento.wav'
archivo_de_salida = 'lapachos_lenta_interpolada.wav'
interpolacion_con_filtro(archivo_de_entrada, archivo_de_salida)
