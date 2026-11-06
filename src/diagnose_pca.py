"""
Diagnóstico del espacio PCA: qué mide cada componente y si separa géneros.

Responde tres preguntas, usando el género (query_tags) SOLO para evaluar,
nunca para construir el espacio:

1. ¿Qué features dominan cada componente? (loadings)
2. ¿Qué componentes separan géneros? Varianza explicada vs. separación (η²)
   lado a lado: PCA maximiza la primera, no la segunda.
3. ¿Los vecinos más cercanos (coseno, igual que similarity.py) son del mismo
   género? Pureza top-k contra la pureza esperada al azar.

Uso:
    python src/diagnose_pca.py [--components 10] [--top 5] [--loadings 4]

Salida: resumen en consola + outputs/pca_diagnostics.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from similarity import build_pipeline, load_features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
LABEL_COL = "query_tags"

# Mismos colores que visualize_track.py (validados para daltonismo). El color
# sigue al género en orden alfabético, así no cambia entre corridas. El marcador
# es una segunda codificación para no depender solo del color.
SERIES = ["#2a78d6", "#eb6834", "#1f9e74", "#a35fd0"]
MARKERS = ["o", "^", "s", "D"]
INK, INK_MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e1"


def separation_by_component(X_pca: np.ndarray, labels: np.ndarray):
    """
    η² por componente: qué fracción de la varianza de esa componente se explica
    por el género (varianza entre grupos / varianza total). 0 = el género no
    dice nada sobre la posición en ese eje; 1 = el eje es el género.
    Funciona con cualquier cantidad de géneros. Con exactamente dos, también
    devuelve la d de Cohen (distancia entre medias en desvíos estándar), que es
    más fácil de leer: d≈0.8 ya se considera un efecto grande.
    """
    classes = np.unique(labels)
    eta2, cohen_d = [], []
    for j in range(X_pca.shape[1]):
        x = X_pca[:, j]
        ss_total = ((x - x.mean()) ** 2).sum()
        ss_between = sum(
            (labels == c).sum() * (x[labels == c].mean() - x.mean()) ** 2 for c in classes
        )
        eta2.append(ss_between / ss_total)
        if len(classes) == 2:
            a, b = x[labels == classes[0]], x[labels == classes[1]]
            pooled = np.sqrt(
                ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2)
            )
            cohen_d.append(abs(a.mean() - b.mean()) / pooled)
    return np.array(eta2), (np.array(cohen_d) if cohen_d else None)


def neighbor_purity(X_pca: np.ndarray, labels: np.ndarray, k: int, metric: str = "cosine"):
    """
    Para cada track, fracción de sus k vecinos más cercanos (excluyéndose a sí
    mismo) que comparten su género. Por defecto usa coseno, igual que
    similarity.py; con metric="euclidean" sirve para espacios como UMAP, donde
    el origen de coordenadas no significa nada y el coseno no tiene sentido.

    El baseline al azar no es 1/n_géneros: si elijo k vecinos al azar entre
    los otros n-1 tracks, la pureza esperada es (n_mismo_género - 1) / (n - 1).
    Con 15+15 tracks da 14/29 ≈ 0.48, no 0.5.
    """
    if metric == "cosine":
        sims = cosine_similarity(X_pca)
    elif metric == "euclidean":
        sims = -euclidean_distances(X_pca)  # más cerca = más similar
    else:
        raise ValueError(f"metric desconocida: {metric}")
    np.fill_diagonal(sims, -np.inf)
    neighbors = np.argsort(-sims, axis=1)[:, :k]
    purity = (labels[neighbors] == labels[:, None]).mean(axis=1)

    n = len(labels)
    counts = pd.Series(labels).value_counts()
    baseline = np.array([(counts[lab] - 1) / (n - 1) for lab in labels])
    return purity, baseline


def print_loadings(pca, feature_cols, n_top: int):
    """
    Loadings = pesos de cada feature (ya escalada) en la componente. El signo
    importa: features con signos opuestos se mueven en direcciones contrarias
    sobre ese eje, y eso suele ser lo que le da significado ("brillante vs
    oscuro"). El signo global de una componente es arbitrario.
    """
    print("\nFeatures que dominan cada componente (loading con signo):")
    for i, comp in enumerate(pca.components_):
        top = np.argsort(-np.abs(comp))[:n_top]
        desc = ", ".join(f"{feature_cols[j]} {comp[j]:+.2f}" for j in top)
        print(f"  PC{i + 1:<2} ({pca.explained_variance_ratio_[i]:5.1%})  {desc}")


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def scatter_pcs(ax, X_pca, labels, classes, i, j, var_ratio, eta2):
    for c_idx, c in enumerate(classes):
        m = labels == c
        ax.scatter(
            X_pca[m, i], X_pca[m, j], s=55, color=SERIES[c_idx], marker=MARKERS[c_idx],
            edgecolors="white", linewidths=1.5, label=c, zorder=3,
        )
    ax.set_xlabel(f"PC{i + 1}  ({var_ratio[i]:.0%} var · η² {eta2[i]:.2f})", color=INK, fontsize=10)
    ax.set_ylabel(f"PC{j + 1}  ({var_ratio[j]:.0%} var · η² {eta2[j]:.2f})", color=INK, fontsize=10)
    style_axis(ax)


def plot(X_pca, labels, var_ratio, eta2, out_path: Path):
    classes = sorted(np.unique(labels))
    best = np.argsort(-eta2)[:2]  # las dos componentes que más separan géneros
    best = np.sort(best)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor="white")

    # (a) Varianza explicada vs separación de géneros, por componente.
    ax = axes[0]
    pcs = np.arange(1, len(var_ratio) + 1)
    w = 0.38
    ax.bar(pcs - w / 2 - 0.01, var_ratio, w, color="#b8b8b3", label="Varianza explicada")
    ax.bar(pcs + w / 2 + 0.01, eta2, w, color=INK, label="η² (separación por género)")
    ax.set_xticks(pcs, [f"PC{p}" for p in pcs], fontsize=8)
    ax.set_ylabel("Fracción (0–1)", color=INK, fontsize=10)
    ax.set_title("PCA ordena por varianza, no por género", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)
    ax.grid(axis="x", visible=False)

    # (b) El plano que muestra cualquier tutorial: PC1×PC2.
    scatter_pcs(axes[1], X_pca, labels, classes, 0, 1, var_ratio, eta2)
    axes[1].set_title("PC1 × PC2 (máxima varianza)", color=INK, fontsize=11, loc="left")
    axes[1].legend(frameon=False, fontsize=9, title=LABEL_COL, title_fontsize=9)

    # (c) El plano de las dos componentes que más separan géneros.
    i, j = best
    scatter_pcs(axes[2], X_pca, labels, classes, i, j, var_ratio, eta2)
    axes[2].set_title(f"PC{i + 1} × PC{j + 1} (máxima separación)", color=INK, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=10, help="Componentes de PCA")
    parser.add_argument("--top", type=int, default=5, help="k vecinos para la pureza")
    parser.add_argument("--loadings", type=int, default=4, help="Features a mostrar por componente")
    args = parser.parse_args()

    df, X, feature_cols = load_features()
    labels = df[LABEL_COL].to_numpy()
    print(f"Catálogo: {len(df)} tracks, {len(feature_cols)} features · "
          + ", ".join(f"{k}={v}" for k, v in df[LABEL_COL].value_counts().items()))

    X_pca, _, pca = build_pipeline(X, n_components=args.components)
    var_ratio = pca.explained_variance_ratio_
    print_loadings(pca, feature_cols, args.loadings)

    eta2, cohen_d = separation_by_component(X_pca, labels)
    print("\nSeparación por género en cada componente:")
    for i in np.argsort(-eta2):
        d = f" · d={cohen_d[i]:.2f}" if cohen_d is not None else ""
        print(f"  PC{i + 1:<2} η²={eta2[i]:.2f}{d}  (varianza {var_ratio[i]:.1%})")

    purity, baseline = neighbor_purity(X_pca, labels, args.top)
    print(f"\nPureza top-{args.top} (vecinos del mismo género):")
    print(f"  global: {purity.mean():.2f}  vs  {baseline.mean():.2f} al azar")
    for lab in sorted(np.unique(labels)):
        m = labels == lab
        print(f"  {lab:<12} {purity[m].mean():.2f}  vs  {baseline[m].mean():.2f}")

    worst = np.argsort(purity)[:5]
    print("\nTracks con menor pureza (candidatos a 'atípicos' de su género):")
    for idx in worst:
        r = df.iloc[idx]
        print(f"  {purity[idx]:.2f}  [{r[LABEL_COL]}] {r['artist']} - {r['title']}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "pca_diagnostics.png"
    plot(X_pca, labels, var_ratio, eta2, out_path)
    print(f"\nFigura: {out_path}")


if __name__ == "__main__":
    main()
