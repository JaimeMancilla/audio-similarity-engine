"""
PCA vs UMAP: ¿qué reducción de dimensionalidad deja más cerca a los tracks
parecidos?

Las dos técnicas responden preguntas distintas:

- **PCA** es lineal y global: busca los ejes de máxima varianza de toda la
  nube de puntos. Cada componente es una combinación de features que se puede
  leer (los loadings). Es determinística: misma entrada, misma salida.
- **UMAP** es no lineal y local: arma un grafo de vecinos en el espacio
  original (cada punto con sus `n_neighbors` más cercanos) y busca una
  disposición en pocas dimensiones que conserve ese grafo. Respeta mucho mejor
  quién es vecino de quién, pero los ejes no significan nada, las distancias
  entre grupos lejanos tampoco, y el resultado cambia con la semilla y con
  `n_neighbors`.

Para un recomendador lo que importa son los vecinos, así que acá se comparan
por dos métricas, ambas sobre los k vecinos más cercanos:

1. **Pureza top-k** contra el género (query_tags). El género se usa SOLO para
   evaluar, nunca para construir el espacio.
2. **Trustworthiness** (sklearn): qué tan fiel es el espacio reducido a los
   vecindarios del espacio original escalado. 1.0 = ningún vecino "intruso"
   que no estaba cerca en el original. No usa el género.

UMAP se corre con varias semillas y se reporta media ± desvío: con 30 tracks,
una sola corrida puede dar un número que es pura suerte.

Uso:
    python src/compare_reduction.py [--top 5] [--seeds 10]

Salida: tabla en consola + outputs/pca_vs_umap.png
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler

from diagnose_pca import INK, INK_MUTED, MARKERS, SERIES, neighbor_purity, style_axis
from similarity import load_features

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
LABEL_COL = "query_tags"


def umap_embed(X_scaled: np.ndarray, n_components: int, n_neighbors: int, seed: int):
    # Import tardío: umap importa numba y tarda unos segundos en compilar; así
    # el resto del proyecto (y los tests) no paga ese costo.
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        # Mismo criterio de similitud que el pipeline: coseno sobre features
        # escaladas. Así la diferencia que medimos es PCA vs UMAP, no
        # coseno vs euclidiana en la entrada.
        metric="cosine",
        random_state=seed,
    )
    with warnings.catch_warnings():
        # Con random_state fijo UMAP avisa que desactiva el paralelismo. Con 30
        # puntos no importa, y preferimos reproducibilidad.
        warnings.filterwarnings("ignore", message=".*n_jobs.*")
        return reducer.fit_transform(X_scaled)


def evaluate(X_scaled, X_emb, labels, k, metric):
    purity, baseline = neighbor_purity(X_emb, labels, k, metric=metric)
    # trustworthiness busca vecinos en el espacio original con `metric` y en el
    # embebido siempre con euclidiana. Para PCA (centrado) la diferencia con
    # coseno es menor; para UMAP la euclidiana es la correcta.
    trust = trustworthiness(X_scaled, X_emb, n_neighbors=k, metric="cosine")
    return purity.mean(), baseline.mean(), trust


def run_comparison(X_scaled, labels, k, seeds, umap_neighbors):
    """
    Devuelve una lista de filas {name, purity, purity_std, trust, trust_std}.
    Los métodos determinísticos (sin reducir, PCA) tienen std 0.
    """
    rows = []

    p, baseline, t = evaluate(X_scaled, X_scaled, labels, k, "cosine")
    rows.append(dict(name="Sin reducir (44-d)", purity=p, purity_std=0.0, trust=t, trust_std=0.0))

    for n_comp in (10, 2):
        X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)
        p, _, t = evaluate(X_scaled, X_pca, labels, k, "cosine")
        tag = " (pipeline actual)" if n_comp == 10 else ""
        rows.append(dict(name=f"PCA {n_comp}{tag}", purity=p, purity_std=0.0, trust=t, trust_std=0.0))

    for n_comp in (10, 2):
        for nn in umap_neighbors:
            ps, ts = [], []
            for seed in range(seeds):
                X_umap = umap_embed(X_scaled, n_comp, nn, seed)
                p, _, t = evaluate(X_scaled, X_umap, labels, k, "euclidean")
                ps.append(p)
                ts.append(t)
            rows.append(dict(
                name=f"UMAP {n_comp} (nn={nn})",
                purity=np.mean(ps), purity_std=np.std(ps),
                trust=np.mean(ts), trust_std=np.std(ts),
            ))

    return rows, baseline


def scatter(ax, X2, labels, classes, title, xlabel, ylabel):
    for c_idx, c in enumerate(classes):
        m = labels == c
        ax.scatter(
            X2[m, 0], X2[m, 1], s=55, color=SERIES[c_idx], marker=MARKERS[c_idx],
            edgecolors="white", linewidths=1.5, label=c, zorder=3,
        )
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    style_axis(ax)


def plot(X_scaled, labels, rows, baseline, k, umap_nn, out_path):
    classes = sorted(np.unique(labels))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1, 1, 1.25]))

    pca2 = PCA(n_components=2, random_state=42).fit(X_scaled)
    X_pca2 = pca2.transform(X_scaled)
    vr = pca2.explained_variance_ratio_
    scatter(axes[0], X_pca2, labels, classes, "PCA 2-d (lineal, ejes interpretables)",
            f"PC1 ({vr[0]:.0%} var)", f"PC2 ({vr[1]:.0%} var)")
    axes[0].legend(frameon=False, fontsize=9, title=LABEL_COL, title_fontsize=9)

    X_umap2 = umap_embed(X_scaled, 2, umap_nn, seed=0)
    scatter(axes[1], X_umap2, labels, classes, f"UMAP 2-d (nn={umap_nn}, semilla 0)",
            "UMAP-1 (sin unidades)", "UMAP-2 (sin unidades)")

    # Pureza por método, con barras de error = desvío entre semillas.
    ax = axes[2]
    names = [r["name"] for r in rows][::-1]
    vals = np.array([r["purity"] for r in rows])[::-1]
    errs = np.array([r["purity_std"] for r in rows])[::-1]
    colors = [INK if n.startswith("PCA 10") else
              (SERIES[0] if n.startswith("UMAP") else "#b8b8b3") for n in names]
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=errs, color=colors, height=0.6,
            error_kw=dict(ecolor=INK_MUTED, lw=1, capsize=3))
    ax.axvline(baseline, color=INK_MUTED, ls="--", lw=1)
    ax.text(baseline, -0.75, f" azar {baseline:.2f}", color=INK_MUTED, fontsize=9, va="center")
    for yi, v, e in zip(y, vals, errs):
        # La etiqueta va después de la barra de error para no taparla.
        ax.text(v + e + 0.015, yi, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, len(names) - 0.5)
    ax.set_xlabel(f"Pureza top-{k} (vecinos del mismo género)", color=INK, fontsize=10)
    ax.set_title("¿Qué espacio junta mejor a los del mismo género?", color=INK, fontsize=11, loc="left")
    style_axis(ax)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5, help="k vecinos para pureza y trustworthiness")
    parser.add_argument("--seeds", type=int, default=10, help="Semillas de UMAP por configuración")
    parser.add_argument("--umap-neighbors", type=int, nargs="+", default=[5, 10, 15],
                        help="Valores de n_neighbors a probar en UMAP")
    args = parser.parse_args()

    df, X, feature_cols = load_features()
    labels = df[LABEL_COL].to_numpy()
    print(f"Catálogo: {len(df)} tracks, {len(feature_cols)} features · "
          + ", ".join(f"{k}={v}" for k, v in df[LABEL_COL].value_counts().items()))

    # Mismo escalado que el pipeline. Ver similarity.build_pipeline para el
    # por qué: sin esto, las features en Hz dominan cualquier reducción.
    X_scaled = StandardScaler().fit_transform(X)

    print(f"Corriendo UMAP con {args.seeds} semillas × {len(args.umap_neighbors)} "
          f"valores de n_neighbors × 2 dimensiones (la primera vez numba compila, tarda)...\n")
    rows, baseline = run_comparison(X_scaled, labels, args.top, args.seeds, args.umap_neighbors)

    print(f"{'Método':<26} {'Pureza top-' + str(args.top):>16}   {'Trustworthiness':>17}")
    for r in rows:
        print(f"{r['name']:<26} {r['purity']:>9.2f} ± {r['purity_std']:.2f}   "
              f"{r['trust']:>10.3f} ± {r['trust_std']:.3f}")
    print(f"{'Azar':<26} {baseline:>9.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "pca_vs_umap.png"
    plot(X_scaled, labels, rows, baseline, args.top, umap_nn=10, out_path=out_path)
    print(f"\nFigura: {out_path}")


if __name__ == "__main__":
    main()
