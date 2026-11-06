"""
Motor de similitud: normaliza features, reduce dimensionalidad con PCA,
y busca los tracks más parecidos a uno dado usando similitud coseno.

Uso:
    python src/similarity.py --query "nombre parcial del track o artista" --top 5
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path(__file__).resolve().parent.parent / "data" / "features" / "features.csv"

# Columnas que NO entran al vector de features. Incluye las etiquetas declaradas
# (query_tags, genre) y la duración: el modelo tiene que ser ciego a ellas; se
# usan únicamente después, para juzgar si los resultados tienen sentido.
META_COLS = ["id", "artist", "title", "album", "query_tags", "genre", "duration_sec"]


def load_features():
    # keep_default_na=False: hay un artista que se llama literalmente "NaN" y
    # pandas lo convertiría en valor faltante. Las columnas numéricas no tienen
    # huecos, así que siguen parseándose como float sin problema.
    df = pd.read_csv(FEATURES_PATH, keep_default_na=False)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    return df, X, feature_cols


def build_pipeline(X: np.ndarray, n_components: int = 10):
    """
    Escala las features (media 0, varianza 1) y reduce dimensionalidad con PCA.

    El escalado ANTES de PCA es crítico: sin él, features con rangos grandes
    (como spectral_rolloff, que puede estar en miles de Hz) dominan sobre
    features normalizadas (como chroma, entre 0 y 1), y PCA termina explicando
    varianza que no tiene que ver con similitud musical real.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} componentes explican {explained:.1%} de la varianza total")

    return X_pca, scaler, pca


def find_similar(df: pd.DataFrame, X_pca: np.ndarray, query: str, top_n: int = 5):
    mask = (
        df["title"].str.contains(query, case=False, na=False)
        | df["artist"].str.contains(query, case=False, na=False)
    )
    matches = df[mask]

    if matches.empty:
        print(f"No se encontró ningún track que coincida con '{query}'")
        return None

    query_idx = matches.index[0]
    query_track = df.loc[query_idx]
    print(f"\nBuscando similares a: {query_track['artist']} - {query_track['title']}\n")

    sims = cosine_similarity(X_pca[query_idx].reshape(1, -1), X_pca)[0]
    df_result = df.copy()
    df_result["similarity"] = sims
    df_result = df_result[df_result.index != query_idx].sort_values("similarity", ascending=False)

    return df_result.head(top_n)[["artist", "title", "album", "similarity"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Texto a buscar en artista/título")
    parser.add_argument("--top", type=int, default=5, help="Cantidad de recomendaciones")
    parser.add_argument("--components", type=int, default=10, help="Componentes de PCA")
    args = parser.parse_args()

    df, X, feature_cols = load_features()
    print(f"Catálogo cargado: {len(df)} tracks, {len(feature_cols)} features crudas\n")

    X_pca, scaler, pca = build_pipeline(X, n_components=args.components)
    results = find_similar(df, X_pca, args.query, args.top)

    if results is not None:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
