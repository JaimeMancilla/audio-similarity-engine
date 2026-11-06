"""
Tests de extracción sobre señales sintéticas con propiedades conocidas.

Un tono puro y ruido blanco son los dos extremos del espectro: el tono
concentra toda la energía en una frecuencia, el ruido la reparte pareja. Por eso
sirven para verificar que cada feature mide lo que dice medir, no solo que el
código corre.
"""
import numpy as np
import pytest
import soundfile as sf

from extract_features import N_MFCC, SAMPLE_RATE, extract_features_from_file

DURATION = 5.0  # segundos: suficiente para beat_track, rápido de procesar


def _write(tmp_path, name, y):
    path = tmp_path / name
    sf.write(path, y.astype(np.float32), SAMPLE_RATE)
    return path


@pytest.fixture(scope="module")
def signals(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("audio")
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    rng = np.random.default_rng(0)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)  # La 440 Hz
    noise = 0.5 * rng.uniform(-1, 1, t.size)
    return {
        "sine": extract_features_from_file(_write(tmp, "sine.wav", sine)),
        "noise": extract_features_from_file(_write(tmp, "noise.wav", noise)),
        "quiet_sine": extract_features_from_file(_write(tmp, "quiet.wav", sine * 0.1)),
    }


def test_vector_de_tamano_fijo(signals):
    feats = signals["sine"]
    # 13 MFCC × (media + std) + 12 chroma + centroid (2) + rolloff + zcr + tempo + rms
    assert len(feats) == 2 * N_MFCC + 12 + 2 + 1 + 1 + 1 + 1 == 44
    assert all(np.isfinite(v) for v in feats.values())


def test_mismo_set_de_columnas_para_cualquier_audio(signals):
    # Si dos tracks dieran columnas distintas, el CSV quedaría con huecos.
    assert signals["sine"].keys() == signals["noise"].keys()


def test_ruido_es_mas_brillante_que_un_tono(signals):
    # 440 Hz puro: el centroide queda cerca de 440. Ruido blanco: cerca de la
    # mitad de la banda (Nyquist/2 ≈ 5.5 kHz).
    assert signals["sine"]["spectral_centroid_mean"] < 1000
    assert signals["noise"]["spectral_centroid_mean"] > 3000
    assert signals["noise"]["spectral_rolloff_mean"] > signals["sine"]["spectral_rolloff_mean"]


def test_zcr_del_ruido_es_mucho_mayor(signals):
    # Un seno de 440 Hz cruza cero 880 veces por segundo → ZCR ≈ 880/22050 ≈ 0.04.
    assert signals["sine"]["zcr_mean"] == pytest.approx(880 / SAMPLE_RATE, rel=0.1)
    assert signals["noise"]["zcr_mean"] > 5 * signals["sine"]["zcr_mean"]


def test_chroma_del_la_domina_en_un_la(signals):
    # chroma_1 = Do ... chroma_10 = La. Un La 440 Hz puro debe dominar ese bin.
    chroma = [signals["sine"][f"chroma_{i}_mean"] for i in range(1, 13)]
    assert int(np.argmax(chroma)) == 9


def test_rms_escala_con_el_volumen(signals):
    ratio = signals["quiet_sine"]["rms_mean"] / signals["sine"]["rms_mean"]
    assert ratio == pytest.approx(0.1, rel=0.05)


def test_mfcc_1_mide_energia_y_el_resto_timbre(signals):
    # MFCC 1 ~ log-energía: baja al bajar el volumen. Los MFCC superiores
    # describen la forma del espectro, que no cambia con el volumen.
    loud, quiet = signals["sine"], signals["quiet_sine"]
    assert quiet["mfcc_1_mean"] < loud["mfcc_1_mean"]
    for i in range(2, 6):
        assert quiet[f"mfcc_{i}_mean"] == pytest.approx(loud[f"mfcc_{i}_mean"], abs=5)
