"""
Clustering no supervisado: ¿emergen grupos naturales en el espacio acústico
sin decirle al modelo qué es un género?

Dos algoritmos con supuestos opuestos:

- **KMeans** asume k grupos "redondos" de tamaño parecido y obliga a cada
  track a pertenecer a uno. Hay que elegir k; acá se elige por silhouette
  (qué tan cerca está cada punto de su cluster vs. del cluster vecino, de -1 a
  1). No mira el género.
- **HDBSCAN** busca regiones densas, elige solo la cantidad de clusters y puede
  dejar puntos como "ruido" (-1) si no pertenecen claramente a ningún grupo.
  Con catálogos chicos suele marcar mucho ruido. Eso es información, no un
  error: dice que no hay grupos densos claros.

Ambos corren sobre el mismo espacio que el recomendador (StandardScaler →
PCA(10)), pero con los vectores **normalizados a norma 1**. KMeans y HDBSCAN
usan distancia euclidiana, y el recomendador usa coseno. Entre vectores de
norma 1, la distancia euclidiana es una función monótona del coseno
(‖a−b‖² = 2 − 2·cos), así que normalizar hace que el clustering vea la misma
noción de "parecido" que el recomendador. A esto se le suele llamar
*spherical k-means*.

El género (query_tags) se usa SOLO al final para evaluar con:
- **ARI** (Adjusted Rand Index): acuerdo entre clusters y géneros, corregido
  por azar. 0 = lo mismo que asignar al azar; 1 = coinciden perfecto.
- **Tabla de contingencia**: cuántos tracks de cada género caen en cada cluster.

Uso:
    python src/cluster.py [--k-max 8] [--min-cluster-size 3]

Salida: resumen en consola + outputs/clusters.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from diagnose_pca import GRID, INK, INK_MUTED, MARKERS, SERIES, style_axis
from similarity import build_pipeline, load_features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
LABEL_COL = "query_tags"
NOISE_COLOR = "#b8b8b3"
# Paleta del proyecto + 4 colores extra, para que clusters distintos nunca
# compartan color (con más de 8 clusters la figura deja de ser legible igual).
CLUSTER_COLORS = SERIES + ["#c9a227", "#c2477a", "#4d6b7f", "#7a9a01"]


def to_cosine_space(X_pca: np.ndarray) -> np.ndarray:
    """L2-normaliza cada fila: euclidiana sobre esto ≡ coseno sobre X_pca."""
    return normalize(X_pca, norm="l2")


def kmeans_sweep(X: np.ndarray, k_values, seed: int = 42):
    """
    Corre KMeans para cada k y devuelve {k: (labels, silhouette)}.
    n_init=20: KMeans arranca de centroides al azar y puede caer en un mínimo
    local; con varias inicializaciones se queda con la mejor (menor inercia).
    """
    results = {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = km.fit_predict(X)
        results[k] = (labels, silhouette_score(X, labels))
    return results


def run_hdbscan(X: np.ndarray, min_cluster_size: int) -> np.ndarray:
    return HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(X)


def contingency(clusters: np.ndarray, genres: np.ndarray) -> pd.DataFrame:
    table = pd.crosstab(
        pd.Series(clusters, name="cluster"), pd.Series(genres, name=LABEL_COL)
    )
    table.index = ["ruido" if c == -1 else f"C{c}" for c in table.index]
    return table


def describe(name: str, clusters: np.ndarray, genres: np.ndarray, df: pd.DataFrame):
    n_noise = int((clusters == -1).sum())
    n_clusters = len(set(clusters)) - (1 if n_noise else 0)
    print(f"\n=== {name}: {n_clusters} clusters"
          + (f", {n_noise} tracks como ruido" if n_noise else "") + " ===")
    print(f"ARI vs género: {adjusted_rand_score(genres, clusters):+.2f}")
    print(contingency(clusters, genres).to_string())
    for c in sorted(set(clusters)):
        members = df[clusters == c]
        tag = "ruido" if c == -1 else f"C{c}"
        names = "; ".join(f"{r.artist} - {r.title}" for r in members.itertuples())
        print(f"  {tag}: {names}")


def plot(X_pca, var_ratio, genres, km_results, best_k, hdb_labels, out_path):
    classes = sorted(np.unique(genres))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), facecolor="white")

    # (a) Silhouette por k.
    ax = axes[0]
    ks = sorted(km_results)
    sils = [km_results[k][1] for k in ks]
    ax.plot(ks, sils, color=INK, marker="o", lw=1.5)
    ax.scatter([best_k], [km_results[best_k][1]], s=120, facecolors="none",
               edgecolors=SERIES[1], linewidths=2, zorder=4)
    ax.set_xlabel("k (cantidad de clusters)", color=INK, fontsize=10)
    ax.set_ylabel("Silhouette (−1 a 1)", color=INK, fontsize=10)
    ax.set_title(f"KMeans: silhouette por k (mejor k={best_k})", color=INK, fontsize=11, loc="left")
    ax.set_xticks(ks)
    # Escala fija 0-0.5 con el umbral de "estructura sustancial" (0.25): un eje
    # autoescalado haría que diferencias de 0.03 parezcan picos dramáticos.
    ax.set_ylim(0, 0.5)
    ax.axhline(0.25, color=INK_MUTED, ls="--", lw=1)
    ax.text(ks[0], 0.255, "umbral de estructura sustancial (0.25)", color=INK_MUTED,
            fontsize=8, va="bottom")
    style_axis(ax)

    # (b) y (c): mismo plano PC1×PC2. El color es el cluster y el marcador es
    # el género, así se ve de un vistazo si los clusters cortan por género.
    def scatter(ax, clusters, title):
        for c_idx, c in enumerate(sorted(set(clusters))):
            color = NOISE_COLOR if c == -1 else CLUSTER_COLORS[c_idx % len(CLUSTER_COLORS)]
            for g_idx, g in enumerate(classes):
                m = (clusters == c) & (genres == g)
                ax.scatter(X_pca[m, 0], X_pca[m, 1], s=60, color=color, marker=MARKERS[g_idx],
                           edgecolors="white", linewidths=1.2, zorder=3)
        ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%} var)", color=INK, fontsize=10)
        ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%} var)", color=INK, fontsize=10)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        style_axis(ax)
        handles = [
            plt.Line2D([], [], color=NOISE_COLOR if c == -1 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                       marker="o", ls="", label="ruido" if c == -1 else f"C{c}")
            for i, c in enumerate(sorted(set(clusters)))
        ] + [
            plt.Line2D([], [], color=INK_MUTED, marker=MARKERS[i], ls="", label=g)
            for i, g in enumerate(classes)
        ]
        ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2)

    # Se grafica k=2 y no el "mejor" k: con dos géneros es la pregunta
    # natural (¿el primer corte cae sobre el género?), y con 30 tracks el
    # mejor silhouette suele ser un artefacto de k grande.
    km_labels = km_results[2][0]
    scatter(axes[1], km_labels,
            f"KMeans k=2 · ARI {adjusted_rand_score(genres, km_labels):+.2f}")
    scatter(axes[2], hdb_labels,
            f"HDBSCAN · ARI {adjusted_rand_score(genres, hdb_labels):+.2f}")

    for ax in axes[1:]:
        ax.grid(True, color=GRID, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=int, default=10, help="Componentes de PCA")
    parser.add_argument("--k-max", type=int, default=8, help="k máximo a probar en KMeans")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Tamaño mínimo de cluster en HDBSCAN")
    args = parser.parse_args()

    df, X, _ = load_features()
    genres = df[LABEL_COL].to_numpy()
    X_pca, _, pca = build_pipeline(X, n_components=args.components)
    X_cos = to_cosine_space(X_pca)

    km_results = kmeans_sweep(X_cos, range(2, args.k_max + 1))
    print("\nKMeans, silhouette por k (más alto = clusters más separados):")
    for k, (labels, sil) in km_results.items():
        print(f"  k={k}  silhouette={sil:.3f}  ARI vs género={adjusted_rand_score(genres, labels):+.2f}")
    best_k = max(km_results, key=lambda k: km_results[k][1])
    best_sil = km_results[best_k][1]
    # Regla práctica (Kaufman & Rousseeuw): silhouette < 0.25 = no hay
    # estructura de clusters sustancial; 0.25-0.5 = débil; > 0.5 = razonable.
    verdict = ("sin estructura sustancial" if best_sil < 0.25 else
               "estructura débil" if best_sil < 0.5 else "estructura razonable")
    print(f"  → mejor silhouette {best_sil:.3f}: {verdict}")
    if best_k == args.k_max:
        # Si el máximo está en el borde, no es un óptimo: con pocos puntos el
        # silhouette tiende a subir con k porque los clusters se vuelven
        # pares/tríos de tracks casi idénticos.
        print(f"  → ojo: el mejor k está en el borde del rango probado (k-max={args.k_max}); "
              "no es un óptimo real")

    describe(f"KMeans k={best_k} (mejor silhouette)", km_results[best_k][0], genres, df)
    if best_k != 2:
        # k=2 es la pregunta natural con dos géneros: ¿el corte en dos cae
        # sobre la frontera de género?
        describe("KMeans k=2", km_results[2][0], genres, df)

    hdb_labels = run_hdbscan(X_cos, args.min_cluster_size)
    describe(f"HDBSCAN (min_cluster_size={args.min_cluster_size})", hdb_labels, genres, df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "clusters.png"
    plot(X_pca, pca.explained_variance_ratio_, genres, km_results, best_k, hdb_labels, out_path)
    print(f"\nFigura: {out_path}")


if __name__ == "__main__":
    main()
