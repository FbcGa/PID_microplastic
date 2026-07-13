"""Tkinter touch UI: live video + config panel. No CV logic here — it only
renders FrameResult snapshots pulled off the worker's queue and forwards
button presses to the callbacks it's given (wired up in main_live.py).

El panel derecho tiene dos vistas intercambiables (pack/pack_forget):
la principal (Inicio|Detener + contadores grandes) y la de calibracion
(margenes sup/inf + capturar fondo + volver), abierta con el icono ⚙.
Tema oscuro: el fondo negro integra el logo institucional (jpg con fondo
negro) y el video sin recuadros visibles.
"""

import queue
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
from PIL import Image, ImageTk

from processing import FrameResult, Mode
from pump import PumpStatus

POLL_MS = 33
CROP_STEP = 5
PANEL_WIDTH = 260

ICONS_DIR = Path(__file__).resolve().parent / "icons"

# Paleta (tema oscuro)
BG = "#000000"
BTN_BG = "#1f1f1f"
BTN_ACTIVE = "#333333"
FG = "#e8e8e8"
FG_DIM = "#666666"
ACCENT_START = "#2e7d32"   # verde Inicio
ACCENT_STOP = "#b71c1c"    # rojo Detener/Salir
FIBER_COLOR = "#ffa726"    # naranja, mismo criterio que CLASS_COLORS
AMORF_COLOR = "#66bb6a"    # verde


@dataclass
class Callbacks:
    on_calibrate: Callable[[], None]
    on_crop_change: Callable[[int, int], None]
    on_capture_background: Callable[[], None]
    on_save_calibration: Callable[[], None]
    on_start: Callable[[], None]
    on_stop: Callable[[], None]
    on_close: Callable[[], None]


def _button(parent, text, command, bg=BTN_BG, fg=FG, font=("Arial", 14, "bold"),
            **kwargs) -> tk.Button:
    """Boton plano estilo tactil con la paleta oscura."""
    return tk.Button(parent, text=text, command=command, font=font, bg=bg,
                     fg=fg, activebackground=BTN_ACTIVE, activeforeground=FG,
                     disabledforeground=FG_DIM, bd=0, relief="flat",
                     highlightthickness=0, **kwargs)


class App:
    def __init__(self, root: tk.Tk, results: "queue.Queue[FrameResult]",
                 callbacks: Callbacks, initial_crop_top: int,
                 initial_crop_bottom: int, background_exists: bool,
                 get_pump_status: Callable[[], PumpStatus]):
        self._root = root
        self._queue = results
        self._cb = callbacks
        self._crop_top = initial_crop_top
        self._crop_bottom = initial_crop_bottom
        self._get_pump_status = get_pump_status
        self._photo: ImageTk.PhotoImage | None = None  # referencia viva, Tkinter la descarta si no

        root.title("Microplasticos - captura en vivo")
        root.configure(bg=BG)
        # Geometria clavada a la pantalla (7'' = 800x480 en la Pi) ademas
        # del fullscreen: sin maxsize/resizable(False) el WM puede seguir
        # agrandando la ventana si algun widget pide mas espacio.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.geometry(f"{screen_w}x{screen_h}+0+0")
        root.resizable(False, False)
        root.maxsize(screen_w, screen_h)
        root.attributes("-fullscreen", True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self._video_w = screen_w - PANEL_WIDTH
        self._video_h = screen_h

        # Frame de tamano fijo (grid_propagate(False)) para que el video no
        # arrastre el tamano de la ventana: sin esto, un Label crece para
        # ajustarse a su imagen, lo que agranda la celda, lo que en el
        # proximo poll se lee como un tamano mayor -> bucle de crecimiento
        # infinito de la ventana.
        video_frame = tk.Frame(root, bg=BG,
                               width=self._video_w, height=self._video_h)
        video_frame.grid(row=0, column=0, sticky="nsew")
        video_frame.grid_propagate(False)

        self._video_label = tk.Label(video_frame, bg=BG)
        # place() no propaga el tamano del hijo al padre (a diferencia de
        # pack/grid), asi que el label puede mostrar cualquier imagen sin
        # afectar el tamano fijo de video_frame.
        self._video_label.place(relx=0.5, rely=0.5, anchor="center")

        panel = tk.Frame(root, width=PANEL_WIDTH, bg=BG)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        # Barra superior: logo institucional a la izquierda, icono de
        # configuracion a la derecha.
        top_bar = tk.Frame(panel, bg=BG)
        top_bar.pack(fill="x", padx=8, pady=(8, 0))

        self._logo_img = self._load_icon("logo.jpg", height=44)
        if self._logo_img is not None:
            tk.Label(top_bar, image=self._logo_img, bg=BG, bd=0).pack(side="left")

        self._gear_img = self._load_icon("config.png", height=32)
        self._btn_config = _button(top_bar, "⚙", self._show_calibration,
                                   font=("Arial", 18), width=3)
        if self._gear_img is not None:
            self._btn_config.config(image=self._gear_img, text="", width=44,
                                    height=44, bg=BG)
        self._btn_config.pack(side="right")

        # Fullscreen sin barra de titulo: el unico camino tactil para
        # cerrar la app es este boton, anclado abajo del panel.
        _button(panel, "Salir", self._cb.on_close, fg="#ff6b6b",
                height=2).pack(side="bottom", fill="x", padx=8, pady=8)

        # --- vista principal ---
        self._main_view = tk.Frame(panel, bg=BG)
        self._main_view.pack(fill="both", expand=True)

        start_stop_row = tk.Frame(self._main_view, bg=BG)
        start_stop_row.pack(fill="x", padx=8, pady=(16, 8))
        self._btn_start = _button(start_stop_row, "Inicio", self._on_start,
                                  bg=ACCENT_START, height=2)
        self._btn_start.config(state="normal" if background_exists else "disabled")
        self._btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._btn_stop = _button(start_stop_row, "Detener", self._on_stop,
                                 bg=ACCENT_STOP, height=2)
        self._btn_stop.config(state="disabled")
        self._btn_stop.pack(side="left", expand=True, fill="x", padx=(4, 0))

        counts_frame = tk.Frame(self._main_view, bg=BG)
        counts_frame.pack(fill="both", expand=True, padx=12, pady=16)
        count_font = ("Arial", 22, "bold")
        self._fiber_label = tk.Label(counts_frame, text="Fibras: 0", bg=BG,
                                     font=count_font, fg=FIBER_COLOR)
        self._fiber_label.pack(anchor="w", pady=6)
        self._amorf_label = tk.Label(counts_frame, text="Amorfas: 0", bg=BG,
                                     font=count_font, fg=AMORF_COLOR)
        self._amorf_label.pack(anchor="w", pady=6)
        tk.Frame(counts_frame, bg="#333333", height=2).pack(fill="x", pady=8)
        self._total_label = tk.Label(counts_frame, text="Total: 0", bg=BG,
                                     font=("Arial", 26, "bold"), fg=FG)
        self._total_label.pack(anchor="w", pady=6)

        pump_font = ("Arial", 14)
        self._caudal_label = tk.Label(counts_frame, text="Caudal: -", bg=BG,
                                      font=pump_font, fg=FG_DIM)
        self._caudal_label.pack(anchor="w", pady=(16, 2))
        self._pump_label = tk.Label(counts_frame, text="Bomba: -", bg=BG,
                                    font=pump_font, fg=FG_DIM)
        self._pump_label.pack(anchor="w")

        # --- vista de calibracion ---
        self._calib_view = tk.Frame(panel, bg=BG)
        self._build_calibration_view(self._calib_view)

        root.after(POLL_MS, self._poll_queue)

    @staticmethod
    def _load_icon(name: str, height: int) -> ImageTk.PhotoImage | None:
        """Icono de icons/ escalado a la altura dada; None si no existe."""
        path = ICONS_DIR / name
        if not path.exists():
            return None
        image = Image.open(path)
        scale = height / image.height
        image = image.resize((max(1, int(image.width * scale)), height))
        return ImageTk.PhotoImage(image)

    def set_start_enabled(self, enabled: bool) -> None:
        self._btn_start.config(state="normal" if enabled else "disabled")

    def _build_calibration_view(self, parent: tk.Frame) -> None:
        step_font = ("Arial", 14, "bold")

        tk.Label(parent, text="Calibracion", bg=BG, fg=FG,
                 font=("Arial", 16, "bold")).pack(anchor="w", padx=8, pady=(16, 8))

        self._top_value = self._build_crop_row(parent, "Sup", step_font,
                                               self._crop_top, "top")
        self._bottom_value = self._build_crop_row(parent, "Inf", step_font,
                                                  self._crop_bottom, "bottom")

        _button(parent, "Guardar fondo", self._cb.on_capture_background,
                height=2).pack(fill="x", padx=8, pady=(16, 8))

        self._calib_caudal_label = tk.Label(parent, text="Caudal: -", bg=BG,
                                            font=("Arial", 12), fg=FG_DIM)
        self._calib_caudal_label.pack(anchor="w", padx=8, pady=(8, 0))

        _button(parent, "Volver", self._show_main,
                height=2).pack(fill="x", padx=8, pady=8)

    def _build_crop_row(self, parent: tk.Frame, title: str, font: tuple,
                        initial: int, which: str) -> tk.Label:
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=8, pady=4)
        tk.Label(row, text=title, font=font, width=4, bg=BG, fg=FG).pack(side="left")
        deltas = {"top": {"top_delta": -CROP_STEP}, "bottom": {"bottom_delta": -CROP_STEP}}
        _button(row, "−", lambda: self._adjust_crop(**deltas[which]),
                font=font, width=3).pack(side="left", padx=2)
        value = tk.Label(row, text=str(initial), font=font, width=4, bg=BG, fg=FG)
        value.pack(side="left")
        plus = {"top": {"top_delta": CROP_STEP}, "bottom": {"bottom_delta": CROP_STEP}}
        _button(row, "+", lambda: self._adjust_crop(**plus[which]),
                font=font, width=3).pack(side="left", padx=2)
        return value

    def _show_calibration(self) -> None:
        self._main_view.pack_forget()
        self._calib_view.pack(fill="both", expand=True)
        self._btn_config.config(state="disabled")
        self._cb.on_calibrate()

    def _show_main(self) -> None:
        self._calib_view.pack_forget()
        self._main_view.pack(fill="both", expand=True)
        self._btn_config.config(state="normal")
        self._cb.on_save_calibration()

    def _adjust_crop(self, top_delta: int = 0, bottom_delta: int = 0) -> None:
        self._crop_top = max(0, self._crop_top + top_delta)
        self._crop_bottom = max(0, self._crop_bottom + bottom_delta)
        self._top_value.config(text=str(self._crop_top))
        self._bottom_value.config(text=str(self._crop_bottom))
        self._cb.on_crop_change(self._crop_top, self._crop_bottom)

    def _on_start(self) -> None:
        self._cb.on_start()
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._btn_config.config(state="disabled")

    def _on_stop(self) -> None:
        self._cb.on_stop()
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._btn_config.config(state="normal")

    def _poll_queue(self) -> None:
        try:
            result = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._render(result)
        self._render_pump_status(self._get_pump_status())
        self._root.after(POLL_MS, self._poll_queue)

    def _render_pump_status(self, status: PumpStatus) -> None:
        caudal_text = f"Caudal: {status.caudal:.0f} ml/min"
        self._caudal_label.config(text=caudal_text)
        self._calib_caudal_label.config(text=caudal_text)
        label = status.membership if status.state == "FUZZY_ACTIVO" else status.state
        self._pump_label.config(text=f"Bomba: {label}")

    def _render(self, result: FrameResult) -> None:
        frame_rgb = cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Escala contra el tamano fijo de video_frame (no el del label): el
        # label sigue el tamano de su imagen, asi que usar su propio
        # winfo_width/height como objetivo crea un bucle de crecimiento.
        scale = min(self._video_w / image.width, self._video_h / image.height)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size)

        self._photo = ImageTk.PhotoImage(image)
        self._video_label.configure(image=self._photo)

        if result.mode == Mode.RUNNING:
            self._fiber_label.config(text=f"Fibras: {result.fibers}")
            self._amorf_label.config(text=f"Amorfas: {result.amorphous}")
            self._total_label.config(text=f"Total: {result.fibers + result.amorphous}")
