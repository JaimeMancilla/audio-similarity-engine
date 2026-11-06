"""Tests del motor de similitud: carga del CSV, pipeline y búsqueda de vecinos."""
import numpy as np
import pandas as pd
import pytest

import similarity
from similarity import META_COLS, build_pipeline, find_similar, load_features


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    """CSV chico con la misma forma que features.csv, en un directorio temporal."""
    rng = np.random.default_rng(0)
    n, n_feat = 12, 6
    X = rng.normal(size=(n, n_feat))
    X[1] = X[0] + 0.01  # el track 1 es casi un duplicado del 0
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_feat)])
    df.insert(0, "id", range(n))
    df.insert(1, "artist", ["NaN"] + [f"Artista {i}" for i in range(1, n)])
    df.insert(2, "title", [f"Tema {i}" for i in range(n)])
    df.insert(3, "album", "")
    df.insert(4, "query_tags", ["a", "b"] * (n // 2))
    df.insert(5, "genre", "")
    df.insert(6, "duration_sec", 180.0)
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(similarity, "FEATURES_PATH", path)
    return df


def test_load_features_excluye_metadata(catalog):
    _, X, feature_cols = load_features()
    assert not set(feature_cols) & set(META_COLS)
    assert feature_cols == [f"f{i}" for i in range(6)]
    assert X.dtype.kind == "f"


def test_artista_llamado_nan_no_se_pierde(catalog):
    # pandas por defecto convierte el texto "NaN" en valor faltante.
    df, _, _ = load_features()
    assert df.loc[0, "artist"] == "NaN"


def test_escalado_hace_al_pipeline_invariante_a_unidades():
    """
    El bug silencioso que el pipeline previene: si una feature viene en
    unidades grandes (Hz vs 0-1), sin escalar dominaría PCA. Con StandardScaler,
    multiplicar una columna por 1000 no debe cambiar nada.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 5))
    X_big = X.copy()
    X_big[:, 0] *= 1000

    a, _, _ = build_pipeline(X, n_components=3)
    b, _, _ = build_pipeline(X_big, n_components=3)
    # El signo de cada componente de PCA es arbitrario: comparar en valor absoluto.
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-8)


def test_pca_sin_escalar_si_seria_dominado():
    """Contraprueba del test anterior: muestra por qué el escalado importa."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 5))
    X[:, 0] *= 1000
    pca = PCA(n_components=1).fit(X)
    assert abs(pca.components_[0, 0]) > 0.99  # PC1 = casi solo la columna grande


def test_n_components_se_recorta_al_maximo_posible():
    X = np.random.default_rng(2).normal(size=(4, 8))
    X_pca, _, pca = build_pipeline(X, n_components=10)
    assert X_pca.shape == (4, 4)


def test_find_similar_devuelve_el_casi_duplicado_primero(catalog):
    df, X, _ = load_features()
    X_pca, _, _ = build_pipeline(X, n_components=5)
    res = find_similar(df, X_pca, "Tema 0", top_n=3)
    assert res.iloc[0]["title"] == "Tema 1"
    assert res.iloc[0]["similarity"] > 0.99
    assert "Tema 0" not in res["title"].values  # no se recomienda a sí mismo
    assert res["similarity"].is_monotonic_decreasing


def test_find_similar_sin_coincidencias(catalog):
    df, X, _ = load_features()
    X_pca, _, _ = build_pipeline(X, n_components=5)
    assert find_similar(df, X_pca, "no existe este tema") is None
