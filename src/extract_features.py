"""
Extrae características acústicas de cada track usando Librosa.

Por cada archivo de audio se calcula un vector de features fijo (no depende
de la duración del track) agregando estadísticas (media, desvío estándar)
sobre las series temporales de cada descriptor:

- MFCCs (13 coeficientes): textura/timbre
- Chroma (12 bins): contenido armónico/tonal
- Spectral centroid: "brillo" del sonido
- Spectral rolloff: energía en altas frecuencias
- Zero-crossing rate: percusividad / ruido
- Tempo (BPM): velocidad estimada

Uso:
    python src/extract_features.py
"""
import json
import sys
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # librosa es ruidoso con warnings de PySoundFile

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
META_PATH = RAW_DIR / "metadata.json"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "features" / "features.csv"

SAMPLE_RATE = 22050  # estándar en librosa, suficiente para estos descriptores
N_MFCC = 13


def extract_features_from_file(filepath: Path) -> dict:
    """Extrae un vector de features de un único archivo de audio."""
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    features = {}

    # --- Timbre / textura ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))

    # --- Armonía / tonalidad ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma.shape[0]):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))

    # --- Brillo espectral ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # --- Energía en altas frecuencias ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))

    # --- Percusividad / ruido ---
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))

    # --- Tempo ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo_bpm"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # --- Energía RMS (volumen percibido) ---
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))

    return features


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # títulos no latinos en consola Windows

    if not META_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {META_PATH}. Corré primero src/fetch_jamendo.py "
            "(o colocá tus propios .mp3/.wav en data/raw/ y un metadata.json a mano)."
        )

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    rows = []

    for i, item in enumerate(metadata, 1):
        filepath = RAW_DIR / item["file"]
        if not filepath.exists():
            print(f"  [{i}/{len(metadata)}] SALTADO (no existe el archivo): {item['file']}")
            continue

        print(f"  [{i}/{len(metadata)}] Extrayendo features: {item['artist']} - {item['title']}")
        try:
            feats = extract_features_from_file(filepath)
        except Exception as e:
            print(f"    ERROR procesando {item['file']}: {e}")
            continue

        row = {
            "id": item["id"],
            "artist": item["artist"],
            "title": item["title"],
            "album": item.get("album", ""),
            # Etiquetas declaradas: viajan en el CSV SOLO para validar resultados.
            # similarity.py las excluye del vector de features vía META_COLS.
            "query_tags": item.get("query_tags", ""),
            "genre": "|".join(item.get("genre", [])),
            "duration_sec": item.get("duration_sec"),
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nListo. {len(df)} tracks procesados.")
    print(f"Features guardadas en {OUT_PATH}")
    print(f"\nColumnas ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
