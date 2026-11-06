"""
Tests de las métricas de evaluación (diagnose_pca, cluster) y de la resolución
del client_id. Se usan datos con respuesta conocida: si los grupos están
perfectamente separados, la pureza tiene que dar 1, η² cerca de 1, ARI 1.
"""
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

import fetch_jamendo
from cluster import kmeans_sweep, to_cosine_space
from diagnose_pca import neighbor_purity, separation_by_component


@pytest.fixture
def two_blobs():
    """Dos grupos de 10 puntos en direcciones opuestas: separación perfecta."""
    rng = np.random.default_rng(0)
    a = rng.normal([5, 0, 0], 0.3, size=(10, 3))
    b = rng.normal([-5, 0, 0], 0.3, size=(10, 3))
    X = np.vstack([a, b])
    labels = np.array(["a"] * 10 + ["b"] * 10)
    return X, labels


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_pureza_perfecta_con_grupos_separados(two_blobs, metric):
    X, labels = two_blobs
    purity, _ = neighbor_purity(X, labels, k=5, metric=metric)
    assert np.all(purity == 1.0)


def test_baseline_de_pureza_no_es_uno_sobre_n_generos(two_blobs):
    # 10 + 10 tracks: al azar, (10 - 1) / (20 - 1) = 9/19, no 0.5.
    X, labels = two_blobs
    _, baseline = neighbor_purity(X, labels, k=5)
    assert baseline == pytest.approx(np.full(20, 9 / 19))


def test_pureza_no_se_cuenta_a_si_mismo():
    # Con k=1 y todos los puntos distintos entre sí, el vecino nunca es uno mismo.
    X = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    labels = np.array(["a", "a", "b", "b"])
    purity, _ = neighbor_purity(X, labels, k=1)
    assert np.all(purity == 1.0)


def test_metrica_desconocida():
    with pytest.raises(ValueError):
        neighbor_purity(np.eye(3), np.array(["a", "b", "c"]), k=1, metric="manhattan")


def test_eta2_y_cohen_d(two_blobs):
    X, labels = two_blobs
    eta2, d = separation_by_component(X, labels)
    assert eta2[0] > 0.95  # el eje 0 es el que separa
    assert eta2[1] < 0.3 and eta2[2] < 0.3  # los otros son ruido
    assert d[0] > 10


def test_espacio_coseno_euclidiana_equivale_a_coseno():
    # Entre vectores de norma 1: ‖a−b‖² = 2 − 2·cos(a, b).
    X = np.random.default_rng(3).normal(size=(6, 4))
    Xn = to_cosine_space(X)
    d2 = ((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(d2, 2 - 2 * cosine_similarity(X), atol=1e-10)


def test_kmeans_recupera_grupos_obvios(two_blobs):
    X, labels = two_blobs
    results = kmeans_sweep(to_cosine_space(X), [2, 3])
    km_labels, sil = results[2]
    assert adjusted_rand_score(labels, km_labels) == 1.0
    assert sil > results[3][1]  # k=2 es mejor que k=3 cuando hay 2 grupos


class TestResolveClientId:
    """Nunca se lee el .env real: ROOT_DIR apunta a un directorio temporal."""

    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_jamendo, "ROOT_DIR", tmp_path)
        monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)
        self.root = tmp_path

    def test_argumento_tiene_prioridad(self, monkeypatch):
        monkeypatch.setenv("JAMENDO_CLIENT_ID", "desde_env")
        assert fetch_jamendo.resolve_client_id("desde_arg") == "desde_arg"

    def test_variable_de_entorno(self, monkeypatch):
        monkeypatch.setenv("JAMENDO_CLIENT_ID", "desde_env")
        assert fetch_jamendo.resolve_client_id(None) == "desde_env"

    def test_archivo_env_con_comillas(self):
        (self.root / ".env").write_text('OTRA=1\nJAMENDO_CLIENT_ID="desde_archivo"\n')
        assert fetch_jamendo.resolve_client_id(None) == "desde_archivo"

    def test_sin_client_id_corta_con_mensaje(self):
        with pytest.raises(SystemExit, match="client_id"):
            fetch_jamendo.resolve_client_id(None)
