"""
Visualización pedagógica: forma de onda, espectrograma Mel y MFCCs de dos tracks
lado a lado, para VER qué está midiendo cada feature antes de confiar en ellas.

Por cada track se toma un fragmento corto (30 s por defecto, del medio del track)
porque un espectrograma de 5 minutos es ilegible y lo que queremos es comparar
textura, no estructura.

Las tres filas cuentan la cadena tiempo -> frecuencia -> timbre:

1. Forma de onda: amplitud vs tiempo. Es el array crudo que devuelve librosa.load.
   Se ve el volumen y la dinámica, pero NO el contenido de frecuencias.
2. Espectrograma Mel (dB): STFT en ventanas cortas -> energía por frecuencia y
   tiempo -> reagrupado en bandas Mel (escala perceptual, comprime los agudos)
   -> log. Acá sí se ve "de qué está hecho" el sonido.
3. MFCCs: transformada coseno del log-espectro Mel. Comprime la envolvente
   espectral (= timbre) en 13 números por frame. El coeficiente 1 es
   básicamente la energía total y domina la escala; los demás describen forma.

Uso:
    python src/visualize_track.py --tracks "In Motion" "Dewey Square"
    python src/visualize_track.py --tracks "Night Tide" "Sandu" --seconds 20 --offset 45
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patheffects import Normal, Stroke

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # solo escribir a archivo, sin ventana

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
META_PATH = RAW_DIR / "metadata.json"
OUT_DIR = ROOT_DIR / "outputs"

SAMPLE_RATE = 22050  # mismo que extract_features.py, para que lo que ves sea lo que se mide
N_MFCC = 13
N_MELS = 128

# --- Paleta (referencia dataviz: superficie clara, tinta recesiva) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834"]  # slot 1 azul, slot 2 naranja: un color por track, fijo

# Secuencial de un solo tono (azul, claro -> oscuro) para magnitud: espectrograma.
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# Divergente azul <-> rojo con gris neutro al medio para valores con signo: MFCCs.
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1f1f"]
)


def find_track(metadata: list[dict], query: str) -> dict:
    q = query.lower()
    for item in metadata:
        if q in item["title"].lower() or q in item["artist"].lower():
            return item
    raise SystemExit(f"No se encontró ningún track que coincida con '{query}'")


def load_excerpt(item: dict, seconds: float, offset: float | None) -> tuple[np.ndarray, int]:
    """Carga `seconds` de audio. Si no se da offset, toma el fragmento del medio."""
    path = RAW_DIR / item["file"]
    duration = float(item.get("duration_sec") or librosa.get_duration(path=path))
    if offset is None:
        offset = max(0.0, duration / 2 - seconds / 2)
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, offset=offset, duration=seconds)
    return y, sr


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_column(axes, y: np.ndarray, sr: int, item: dict, color: str, mfcc_lim: float):
    ax_wave, ax_spec, ax_mfcc = axes
    halo = [Stroke(linewidth=3, foreground=SURFACE), Normal()]  # legible sobre cualquier fondo

    # Descriptores escalares del MISMO fragmento, para conectar imagen con número.
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # 1) Forma de onda ------------------------------------------------------
    t = np.arange(len(y)) / sr
    ax_wave.plot(t, y, color=color, linewidth=0.4)
    ax_wave.set_xlim(0, t[-1])
    ax_wave.set_ylim(-1, 1)
    ax_wave.set_ylabel("amplitud")
    ax_wave.grid(True, color=GRID, linewidth=0.5)
    ax_wave.set_title(
        f"{item['artist']} — {item['title']}",
        color=INK, fontsize=10, fontweight="bold", loc="left", pad=18,
    )
    ax_wave.text(
        0.0, 1.03,
        f"{item['query_tags']}  ·  centroid {centroid:,.0f} Hz  ·  ZCR {zcr:.3f}  ·  RMS {rms:.3f}",
        transform=ax_wave.transAxes, color=INK_2, fontsize=8, va="bottom",
    )

    # 2) Espectrograma Mel en dB ----------------------------------------------
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)  # 0 dB = pico del fragmento
    img = librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax_spec, cmap=CMAP_SEQ, vmin=-80, vmax=0
    )
    ax_spec.set_ylabel("frecuencia (Mel)")
    ax_spec.set_xlabel("")
    ax_spec.axhline(centroid, color=INK, linewidth=1.2, linestyle="--", path_effects=halo)
    ax_spec.text(
        0.01, centroid, f"centroid {centroid:,.0f} Hz", color=INK, fontsize=7, ha="left", va="bottom",
        transform=ax_spec.get_yaxis_transform(), path_effects=halo,
    )
    plt.colorbar(img, ax=ax_spec, format="%+2.0f dB", pad=0.01).ax.tick_params(colors=MUTED, labelsize=7)

    # 3) MFCCs ----------------------------------------------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    img = librosa.display.specshow(
        mfcc, sr=sr, x_axis="time", ax=ax_mfcc, cmap=CMAP_DIV,
        norm=TwoSlopeNorm(vmin=-mfcc_lim, vcenter=0.0, vmax=mfcc_lim),
    )
    ax_mfcc.set_yticks(range(N_MFCC))
    ax_mfcc.set_yticklabels([str(i + 1) for i in range(N_MFCC)])
    ax_mfcc.set_ylabel("coeficiente MFCC")
    ax_mfcc.set_xlabel("tiempo (s)")
    plt.colorbar(img, ax=ax_mfcc, pad=0.01).ax.tick_params(colors=MUTED, labelsize=7)

    for ax in axes:
        style_axis(ax)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs=2, required=True, metavar="QUERY",
                        help="Dos búsquedas parciales (título o artista), una por columna")
    parser.add_argument("--seconds", type=float, default=30.0, help="Duración del fragmento")
    parser.add_argument("--offset", type=float, default=None,
                        help="Segundo de inicio del fragmento (default: el medio del track)")
    parser.add_argument("--out", default=None, help="Ruta del PNG de salida")
    args = parser.parse_args()

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    items = [find_track(metadata, q) for q in args.tracks]

    fig, axes = plt.subplots(
        3, 2, figsize=(14, 10), facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1, 2, 1.4], "hspace": 0.45, "wspace": 0.25},
    )
    fig.suptitle(
        f"Forma de onda → espectrograma Mel → MFCCs   (fragmento de {args.seconds:.0f} s, {SAMPLE_RATE} Hz)",
        color=INK, fontsize=12, x=0.02, ha="left",
    )

    excerpts = [load_excerpt(item, args.seconds, args.offset) for item in items]

    # Escala de color de MFCC compartida entre columnas, si no no se pueden comparar.
    # El coeficiente 1 (energía) es ~10x más grande que el resto: si dejamos que
    # fije la escala, los otros 12 quedan planos. Escala simétrica robusta sobre
    # los coeficientes 2..13; el 1 satura, y eso es parte de la lección.
    mfcc_lim = max(
        float(np.percentile(np.abs(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)[1:]), 98))
        for y, sr in excerpts
    )

    for col, (item, color, (y, sr)) in enumerate(zip(items, SERIES, excerpts)):
        print(f"Procesando columna {col + 1}: {item['artist']} - {item['title']}")
        plot_column(axes[:, col], y, sr, item, color, mfcc_lim)

    OUT_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"viz_{items[0]['id']}_vs_{items[1]['id']}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
