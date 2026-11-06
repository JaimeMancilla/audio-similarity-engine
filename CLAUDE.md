# CLAUDE.md

Contexto para Claude Code sobre este proyecto. Leer antes de tocar código.

## Qué es este proyecto

Motor de recomendación musical por **similitud de contenido acústico**
(Audio Content-Based Filtering) — no colaborativo. La idea central: dado un
track, encontrar otros tracks del catálogo con textura sonora, tempo y color
similares, analizando el audio crudo con matemática de señales, sin usar
historial de escucha de usuarios ni metadata de género como input del modelo
(el género solo se usa después, para *validar* si los resultados tienen
sentido).

Es un proyecto de **aprendizaje personal** (Data Science + Digital Signal
Processing aplicado a audio) con intención de publicarse como repo abierto.
Prioridad: entender por qué cada paso funciona, no solo que funcione. Preferí
explicaciones sobre atajos, y señalá explícitamente cuando una elección de
diseño tiene trade-offs (por ejemplo PCA vs UMAP, o tamaño de ventana STFT).

El usuario (Jaime) es profesional de datos/ML (pipelines de entity resolution
en su trabajo), pero este proyecto es la primera vez que trabaja en serio con
señales de audio — no asumir conocimiento previo de DSP (FFT, STFT, escala
Mel, cepstrum), pero sí de ML/Python en general.

## Stack

Python 3.13 en `.venv/` (NO el 3.14 del sistema: numba/librosa tardan en
soportar versiones nuevas) · Librosa 1.0 (extracción de features de audio) ·
Scikit-learn (StandardScaler, PCA, similitud coseno, clustering futuro) ·
Pandas/NumPy · matplotlib (visualización) · requests (API de Jamendo) ·
umap-learn (solo en `compare_reduction.py`, import tardío porque numba tarda).
Correr todo con `.venv/Scripts/python.exe src/<script>.py`.

## Estado actual del pipeline

Pipeline completo **corrido por primera vez el 2026-09-21** sobre 30 tracks
(15 `shoegaze` + 15 `jazz`, 163 MB en `data/raw/`, `features.csv` de 30×51).

Scripts en `src/`, en orden de ejecución:

1. **`fetch_jamendo.py`** — descarga tracks CC desde la API de Jamendo.
   `client_id` por `--client-id`, env `JAMENDO_CLIENT_ID` o `.env` en la raíz.
   Usa `fuzzytags` (el filtro `tags` exacto de Jamendo es errático, devuelve 0
   para tags que existen). **Acumula** `metadata.json` entre corridas (dedupe
   por id) y guarda `query_tags` = con qué tag se buscó cada track — es la
   etiqueta de validación, nunca entra al modelo. Se corre una vez por género.
2. **`extract_features.py`** — Librosa sobre cada archivo → vector de 44
   features de tamaño fijo por track (agregación media/std, no series
   temporales): 13 MFCCs (mean+std), 12 chroma (mean), spectral
   centroid (mean+std)/rolloff, ZCR, tempo (BPM), RMS. El CSV lleva además
   `query_tags`, `genre` y `duration_sec` como metadata (~3 s por track).
3. **`similarity.py`** — `StandardScaler` → `PCA(10)` → similitud coseno.
   `META_COLS` excluye explícitamente las etiquetas del vector de features —
   si se agrega una columna de metadata al CSV, hay que agregarla ahí también
   o entra como feature numérica. Lee el CSV con `keep_default_na=False`
   porque hay un artista que se llama literalmente "NaN".
4. **`visualize_track.py`** — pedagógico: forma de onda / espectrograma Mel /
   MFCCs de dos tracks lado a lado (fragmento de 30 s del medio). Salida en
   `outputs/viz_<id>_vs_<id>.png`. Escala de MFCC compartida entre columnas.
5. **`diagnose_pca.py`** — reusa `load_features`/`build_pipeline` de
   `similarity.py`. Imprime loadings por componente, separación por género
   por componente (η², y d de Cohen si hay 2 géneros), pureza top-k contra
   baseline al azar `(n_mismo-1)/(n-1)` y los tracks de menor pureza. Figura
   en `outputs/pca_diagnostics.png`: varianza vs η² por PC, PC1×PC2, y el
   plano de las dos PCs que más separan. Es la vara para medir cambios
   futuros al pipeline. `neighbor_purity` acepta `metric="euclidean"`.
6. **`compare_reduction.py`** — PCA vs UMAP sobre las features escaladas:
   pureza top-k + trustworthiness (sklearn), UMAP con varias semillas y
   `n_neighbors` ∈ {5,10,15}, en 2-d y 10-d. UMAP usa `metric="cosine"` en la
   entrada y euclidiana en el embebido (el origen de UMAP no significa nada).
   Figura en `outputs/pca_vs_umap.png`.

Catálogo de prueba: **shoegaze + jazz**, elegidos por ser polos casi opuestos.
Sanity check que ya pasa: las dos versiones de *Nargo - Dreamwave Reverie*
son vecinas más cercanas con coseno 0.97; *Dewey Square* y *Sandu* (mismo
artista/sesión) 0.93.

## Decisiones de diseño ya tomadas (y por qué)

- **Escalar (StandardScaler) SIEMPRE antes de PCA.** Sin esto, features de
  magnitud grande (spectral_rolloff, en miles de Hz) dominan la varianza
  explicada sobre features ya acotadas (chroma, 0-1), sin que eso refleje
  similitud musical real. Este es el bug silencioso más probable si alguien
  toca el pipeline sin saber esto — no tira error, solo da resultados malos.
- **Agregación estadística por track, no frames.** Cada MFCC/chroma se resume
  a media (y std en el caso de MFCCs) a lo largo del tiempo, para tener un
  vector de longitud fija independiente de la duración del track.
- **Similitud coseno, no euclidiana**, sobre el espacio PCA — más robusta a
  diferencias de magnitud entre vectores ya normalizados.
- **Unidad de análisis doble (pendiente de implementar)**: el usuario quiere
  trabajar tanto a nivel canción como a nivel álbum. Cuando se implemente
  agregación a nivel álbum, probablemente sea centroide o promedio de los
  vectores de canción del álbum — a decidir junto con el usuario, no asumir.

## Trampa conceptual shoegaze vs jazz: hipótesis vs. lo que se midió

**Hipótesis previa** (antes de correr nada): shoegaze separaría "demasiado
fácil" por tener ZCR y spectral centroid altos por la distorsión, y jazz
acústico limpio los tendría bajos — separación trivial "sucio vs limpio".

**Resultado empírico (2026-09-21, 30 tracks): la hipótesis NO se cumple.**

- Centroid medio: shoegaze 1938 Hz vs jazz 1993 Hz. ZCR: 0.084 vs 0.077.
  Prácticamente iguales. Jazz tiene el doble de *dispersión*.
- El "jazz" de Jamendo incluye nujazz, triphop, funk, electropop, bebop —
  platillos, bronces y producción moderna generan tanta alta frecuencia como
  una guitarra distorsionada. *Dewey Square* (bebop, ride constante) tiene
  centroid 3.5 kHz y ZCR 0.19, el doble que *In Motion* (shoegaze).
- El eje "brillo" (centroid/rolloff vs mfcc_2/rms) es **PC3** (12%), no PC1,
  y shoegaze cae del lado *oscuro*. PC1 (24%) es la variabilidad temporal
  del timbre (std de MFCCs 6-12): el muro de sonido es estático, el jazz
  cambia de textura todo el tiempo.
- La componente que mejor separa géneros es PC5 (d=1.11, η²=0.25) con solo 7% de
  varianza. Lección: PCA maximiza varianza, no separación de clases — es no
  supervisado y no sabe qué es un género.
- Pureza top-5 (vecinos con mismo tag): 0.61 vs 0.48 al azar. Hay señal, es
  modesta, y ningún track tiene pureza 0. No hay clusters "perfectos".

**PCA vs UMAP (2026-11-06):** pureza top-5 PCA10 0.61, sin reducir 0.60,
UMAP10 0.58±0.01, UMAP2 0.59±0.05, PCA2 0.49 (= azar). UMAP gana en
trustworthiness (0.956 vs 0.923) pero no en pureza → el cuello de botella son
las features, no la reducción. Se queda PCA(10) en el recomendador. Revisitar
UMAP con catálogo más grande.

Sigue vigente: el tempo de `librosa.beat.beat_track()` es poco confiable en
jazz con swing o ritmo libre — limitación conocida de MIR, no bug.

## Qué falta (roadmap real, no aspiracional)

- [x] Visualización de forma de onda / espectrograma / MFCCs
      (`src/visualize_track.py`, 2026-09-21).
- [x] Diagnóstico de PCA: loadings, η²/d por componente, pureza top-k,
      scatter coloreado por género (`src/diagnose_pca.py`, 2026-11-02).
- [ ] Agregación de features a nivel álbum.
- [ ] Clustering no supervisado (KMeans / HDBSCAN) para ver si emergen
      agrupaciones sin usar género como etiqueta.
- [x] Comparar PCA vs UMAP (`src/compare_reduction.py`, 2026-11-06).
- [ ] Anomaly detection: tracks atípicos dentro de su propio género
      declarado.
- [ ] Tests — el proyecto todavía no tiene ninguno.
- [x] Publicación MVP (2026-11-06): scripts (no notebooks), sin packaging,
      licencia MIT, figuras del README copiadas a `docs/img/` (porque
      `outputs/*.png` está en `.gitignore` — si se regeneran, copiarlas).

## Cómo trabajar en este proyecto

- Explicar el "por qué" de cambios de pipeline, no solo aplicarlos — el
  objetivo explícito es aprendizaje.
- Cuando se toque `extract_features.py`, tener en cuenta que cambiar
  parámetros de Librosa (`n_fft`, `hop_length`, `n_mfcc`) invalida features
  ya extraídas — hay que re-correr el pipeline completo, no mezclar CSVs de
  distintas configuraciones.
- El `client_id` de Jamendo es secreto del usuario (no hardcodear, no
  commitear) — se pasa por `--client-id` o variable de entorno.
- Antes de publicar el repo: revisar que no haya audio descargado (`data/raw/*.mp3`)
  ni credenciales en el historial de git — el audio de Jamendo tiene licencia
  CC pero probablemente no corresponda redistribuirlo dentro del repo; usar
  `.gitignore` para `data/raw/` y `data/features/`. Repo git inicializado
  el 2026-11-06 y publicado en GitHub (público).
- Consola de Windows es cp1252: los scripts hacen
  `sys.stdout.reconfigure(encoding="utf-8")` porque hay títulos en cirílico.
  Para scripts ad-hoc, `PYTHONIOENCODING=utf-8`.
