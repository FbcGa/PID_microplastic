from dataclasses import dataclass

@dataclass(frozen=True)
class TrackerConfig:
    # Max distance (px) between predicted position and detection to match
    max_distance: float = 300.0
    # Frames a track survives without being seen (~0.3 s at 30 FPS)
    max_missed: int = 5
    # Frames seen before a track is confirmed (and counted)
    min_hits: int = 2
    # EMA smoothing of the velocity
    velocity_alpha: float = 0.5


@dataclass(frozen=True)
class BackgroundSegmentationConfig:
    # Umbral sobre |frame_g - fondo_g|: pixel con diferencia por encima = particula
    diff_thresh: int = 10
    # Blur previo al umbral, reduce ruido de sensor
    blur_ksize: int = 3
    # Apertura: quita puntos sueltos de ruido
    open_ksize: int = 1
    # Cierre: une fragmentos rotos de una misma fibra
    close_ksize: int = 5
    # Filtro de tamano (px^2) — provisional hasta la calibracion um -> px
    min_area: float = 350.0
    max_area: float = 8000.0
    # Circularidad = 4*pi*Area/Perimetro^2. Blobs con circularidad >=
    # max_circularity se descartan por ser burbujas.
    max_circularity: float = 0.8
