# PID — Detección y clasificación de microplásticos

Pipeline de procesamiento de imágenes para detectar microplásticos en video
(cámara fija, iluminación controlada) y clasificarlos como **fibra** o
**amorfa** con un Random Forest entrenado sobre descriptores de forma e
intensidad.

## Requisitos

- Python >= 3.13, gestionado con [uv](https://docs.astral.sh/uv/)
- Dependencias: OpenCV, NumPy, SciPy, scikit-learn, matplotlib (ver `pyproject.toml`)

```bash
uv sync
```

## Flujo de trabajo

### 1. Extraer frames de los videos

```bash
uv run get_frames.py                     # videos/ -> frames/, 1 fps
uv run get_frames.py --fps 2             # dos frames por segundo
uv run get_frames.py --every 30          # un frame cada 30 frames
uv run get_frames.py --max-per-video 20  # tope por video
```

Genera `frames/<nombre_video>/frame_*.jpg`.

### 2. Etiquetar partículas (construir el dataset)

```bash
uv run label_particles.py                # frames/ -> dataset.csv
```

Segmenta cada frame con el método de `detector_v2.py` y muestra cada
partícula resaltada para etiquetarla con el teclado:

| Tecla | Acción |
|---|---|
| `f` | fibra |
| `a` | amorfa |
| `s` | omitir (no se guarda) |
| `x` | partículas cruzadas/fusionadas (excluidas; recorte guardado en `merged/`) |
| `q` | salir y guardar |

Cada partícula etiquetada es una fila de `dataset.csv` (vector de
descriptores + etiqueta).

### 3. Entrenar el clasificador

```bash
uv run train_rf.py                       # dataset.csv -> rf_model.joblib
```

Entrena un `RandomForestClassifier` (`class_weight="balanced"`, sin
escalado) y reporta validación cruzada, matriz de confusión e importancia
de features.

### 4. Detectar y clasificar sobre un frame

```bash
uv run detector_v2.py frames/frame2.jpg
uv run detector_v2.py frames/frame2.jpg --green-thresh 100 --min-area 200
uv run detector_v2.py frames/frame2.jpg --no-classify   # solo contornos
```

Si existe `rf_model.joblib`, dibuja cada contorno con el color de su clase
(fibra naranja, amorfa verde); si no, muestra solo la segmentación con el
filtro de área (verde = aceptado, rojo = descartado).

## Métodos de segmentación

- **`detector_v2.py`** — umbral sobre el canal verde (`G < umbral`,
  portado de un script MATLAB validado). Es el método usado por el
  etiquetado y la clasificación actuales.
- **`detector.py`** — oscuridad local: fondo local estimado con box blur,
  binarización de `fondo - frame` y filtros heurísticos de burbujas
  (borde desenfocado, anillo brillante). Modo legacy `background`
  (sustracción contra frame de referencia) disponible vía
  `config.segmentation_mode`. Visor de un frame:
  `uv run detector.py frames/frame2.jpg`.

## Archivos

| Archivo | Rol |
|---|---|
| `config.py` | Parámetros calibrados de detección (calibración Colab, jun 2026) |
| `get_frames.py` | Extracción de frames desde `videos/` |
| `label_particles.py` | Herramienta interactiva de etiquetado |
| `features.py` | Extracción de descriptores — única fuente de verdad del vector de features (entrenamiento e inferencia) |
| `train_rf.py` | Entrenamiento y evaluación del Random Forest |
| `rf_classifier.py` | Clasificador en inferencia (carga `rf_model.joblib`) |
| `detector_v2.py` | Segmentación por canal verde + clasificación RF |
| `detector.py` | Segmentación por oscuridad local + filtros de burbujas |
| `dataset.csv` | Dataset etiquetado (generado) |
| `rf_model.joblib` | Modelo entrenado (generado) |

## Descriptores (20)

Forma: área, perímetro, aspect ratio, circularidad, solidez, extent,
ancho medio, diámetro equivalente, excentricidad, elongación y los 7
momentos de Hu. Intensidad: media, desviación estándar y oscuridad pico.
El orden canónico está en `features.FEATURE_NAMES`.

## Notas

- `main.py` (reproducción de video con tracking) está desactualizado:
  importa `tracker`, `video_source` y `visualizer`, módulos ya eliminados.
  El flujo vigente es el de frames sueltos descrito arriba.
