# PID — Detección y clasificación de microplásticos

Pipeline de procesamiento de imágenes para detectar microplásticos en video
(cámara fija, iluminación controlada) y clasificarlos como **fibra** o
**amorfa** con un Random Forest entrenado sobre descriptores de forma.

## Requisitos

- Python >= 3.13, gestionado con [uv](https://docs.astral.sh/uv/)
- Dependencias: OpenCV, NumPy, SciPy, scikit-learn, matplotlib (ver `pyproject.toml`)

```bash
uv sync
```

## Flujo principal

Detección + clasificación RF + tracking en vivo sobre un video. Requiere un
modelo ya entrenado (`random_forest/rf_model.joblib`; ver
[Entrenar el modelo](#entrenar-el-modelo) si todavía no existe) y una imagen
de fondo (agua limpia, sin partículas).

```bash
uv run main.py video.mp4 --background fondo.jpg
uv run main.py video.mp4 --background fondo.jpg --model random_forest/rf_model.joblib --no-display
```

Cada frame: `detector.py` segmenta (resta contra la imagen de fondo) + el
Random Forest clasifica cada partícula (fibra/amorfa) + el tracker
(`tracker.py`) asigna IDs estables para contar cada partícula una sola vez.
El overlay (contornos, IDs, conteo corriendo) lo dibuja `visualization.py`.

Teclas durante la reproducción: `q` / `ESC` salir, `espacio` pausar/reanudar.

### Detectar y clasificar sobre un solo frame

Para inspeccionar el pipeline de segmentación sin correr un video completo:

```bash
uv run detector.py frames/frame2.jpg --background frames/fondo.jpg
uv run detector.py frames/frame2.jpg --background frames/fondo.jpg --no-classify   # solo contornos
uv run detector.py frames/frame2.jpg --background frames/fondo.jpg --debug-stages  # ver cada etapa del pipeline
```

Si existe `random_forest/rf_model.joblib`, dibuja cada contorno con el color
de su clase (fibra naranja, amorfa verde); si no, muestra solo la
segmentación con el filtro de área (verde = aceptado, rojo = descartado).

**Método de segmentación** (`detector.py`): resta contra una imagen de fondo
real (agua limpia, sin partículas) en el canal verde — `absdiff` + blur +
umbral + apertura/cierre morfológico. Parámetros calibrados en
`config.BackgroundSegmentationConfig`.

## Entrenar el modelo

Construye el dataset etiquetado y entrena el Random Forest que usa el flujo
principal. Pasos en orden:

### 1. Extraer frames de los videos

```bash
uv run tools/get_frames.py                     # videos/ -> frames/, 1 fps
uv run tools/get_frames.py --fps 2             # dos frames por segundo
uv run tools/get_frames.py --every 30          # un frame cada 30 frames
uv run tools/get_frames.py --max-per-video 20  # tope por video
```

Genera `frames/<nombre_video>/frame_*.jpg`.

### 2. Etiquetar partículas (construir el dataset)

```bash
uv run tools/label_particles.py --background frames/fondo.jpg                     # frames/ -> random_forest/dataset.csv
uv run tools/label_particles.py --background frames/fondo.jpg --frames frames/30_20260709_013216   # solo esa carpeta
```

`--frames` acepta cualquier carpeta (busca recursivamente adentro), así que
sirve para etiquetar solo los frames de un video/subcarpeta puntual en vez
de todo `frames/`.

Segmenta cada frame con el método de `detector.py` y muestra cada partícula
resaltada para etiquetarla con el teclado:

| Tecla | Acción |
|---|---|
| `f` | fibra |
| `a` | amorfa |
| `s` | omitir (no se guarda) |
| `x` | partículas cruzadas/fusionadas (excluidas; recorte guardado en `merged/`) |
| `q` | salir y guardar |

Cada partícula etiquetada es una fila de `random_forest/dataset.csv` (vector
de descriptores + etiqueta).

### 3. Entrenar el clasificador

```bash
uv run random_forest/train_rf.py         # random_forest/dataset.csv -> random_forest/rf_model.joblib
```

Entrena un `RandomForestClassifier` (`class_weight="balanced"`, sin
escalado) y reporta validación cruzada agrupada por frame (`GroupKFold`
sobre `source_frame`, así ninguna partícula del frame de entrenamiento
aparece también en el de validación), matriz de confusión e importancia de
features.

**Descriptores (11)** — solo forma pura: perímetro, aspect ratio,
circularidad, solidez, extent, ancho medio, excentricidad, elongación y los
momentos de Hu 1, 2 y 4. Se descartaron area, equivalent_diameter, hu_3/5/6/7
y los descriptores de intensidad (mean_intensity, std_intensity,
peak_darkness) por tener importancia ~0 en el Random Forest entrenado. El
orden canónico está en `features.FEATURE_NAMES`.

## Tools (`tools/`)

Scripts auxiliares que no forman parte del flujo principal en vivo — se usan
para preparar datos, depurar o capturar video.

| Script | Uso |
|---|---|
| `tools/get_frames.py` | Extrae frames de `videos/` (ver [paso 1](#1-extraer-frames-de-los-videos)) |
| `tools/label_particles.py` | Etiquetado interactivo para construir el dataset (ver [paso 2](#2-etiquetar-partículas-construir-el-dataset)) |
| `tools/debug_tracker.py` | `uv run tools/debug_tracker.py video.mp4 --background fondo.jpg` — corre el mismo pipeline que `main.py` y exporta un CSV frame a frame (posiciones, velocidad, votos de clase, hits, conteos) más un video anotado con cada track en vivo. `--no-video` para solo CSV (más rápido). |
| `tools/cut_video.py` | `uv run tools/cut_video.py video.mp4 --start 10 --end 25` — recorta un video a un rango de tiempo (acepta segundos o `mm:ss`/`hh:mm:ss`). |
| `tools/show_grid.py` | `uv run tools/show_grid.py frames/frame_000000.jpg --step 25` — dibuja una cuadrícula de referencia en píxeles sobre un frame, para calibrar `max_distance`, filtros de área, etc. a ojo. |
| `tools/grabar.py` | Graba video desde la cámara vía `rpicam-vid` (Raspberry Pi). Parámetros de captura (fps, shutter, ganancia) se editan como constantes en el archivo. |

## Archivos

| Archivo | Rol |
|---|---|
| `main.py` | Entry point del flujo principal (detección + clasificación + tracking en vivo) |
| `detector.py` | Segmentación por resta contra fondo + clasificación RF |
| `tracker.py` | Tracking de partículas entre frames |
| `visualization.py` | Overlay de contornos, IDs y conteos sobre el frame |
| `features.py` | Extracción de descriptores — única fuente de verdad del vector de features (entrenamiento e inferencia) |
| `config.py` | Parámetros calibrados de detección y tracking (calibración Colab, jun 2026) |
| `utils.py` | Helpers compartidos (recorte de bordes) |
| `random_forest/rf_classifier.py` | Clasificador en inferencia (carga `rf_model.joblib`) |
| `random_forest/train_rf.py` | Entrenamiento y evaluación del Random Forest |
| `random_forest/dataset.csv` | Dataset etiquetado (generado) |
| `random_forest/rf_model.joblib` | Modelo entrenado (generado) |
| `tools/` | Scripts auxiliares fuera del flujo principal (ver [Tools](#tools-tools)) |
