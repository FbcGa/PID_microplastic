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
from arduino.arduino import Membership, PumpState, PumpStatus

POLL_MS = 33
CROP_STEP = 5

# Layout fijo a la pantalla de 7'' de la Pi (800x480): mismas proporciones
# en la PC de desarrollo, y presupuesto de alto conocido para que todos
# los botones entren siempre en el panel.
WINDOW_W = 800
WINDOW_H = 480
PANEL_WIDTH = 240

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

# Texto amigable para el estado de la bomba: unica fuente de verdad, para
# no imprimir el valor crudo del protocolo (MEDIA, RAMPA_STOP) en la UI.
MEMBERSHIP_LABELS = {
    Membership.MUY_POCAS: "Muy pocas",
    Membership.POCAS: "Pocas",
    Membership.MEDIA: "Media",
    Membership.MUCHAS: "Muchas",
    Membership.NONE: "-",
}
STATE_LABELS = {
    PumpState.OFF: "Apagada",
    PumpState.LIMPIEZA: "Limpieza",
    PumpState.RAMPA_SUBIDA: "Arrancando",
    PumpState.FUZZY_ACTIVO: "Fuzzy",
    PumpState.RAMPA_STOP: "Deteniendo",
}
# Mismo criterio que STATE_LABELS/MEMBERSHIP_LABELS: unica fuente de verdad
# para el color, asi Bomba/Membresia se leen de un vistazo en el panel.
STATE_COLORS = {
    PumpState.OFF: FG_DIM,
    PumpState.LIMPIEZA: FG_DIM,
    PumpState.RAMPA_SUBIDA: "#ffb74d",
    PumpState.FUZZY_ACTIVO: ACCENT_START,
    PumpState.RAMPA_STOP: "#ffb74d",
}
MEMBERSHIP_COLORS = {
    Membership.MUY_POCAS: FG_DIM,
    Membership.POCAS: "#9ccc65",
    Membership.MEDIA: "#ffca28",
    Membership.MUCHAS: "#ef5350",
    Membership.NONE: FG_DIM,
}


@dataclass
class Callbacks:
    on_calibrate: Callable[[], None]
    on_crop_change: Callable[[int, int], None]
    on_capture_background: Callable[[], None]
    on_save_calibration: Callable[[], None]
    on_start: Callable[[], None]
    on_stop: Callable[[], None]
    on_reset: Callable[[], None]
    on_close: Callable[[], None]


# Tamanos de fuente NEGATIVOS = pixeles (Tk): con puntos, Windows los
# escala segun el DPI (125%/150%) y el layout de 800x480 se desborda en
# la PC de desarrollo; en pixeles rinde identico en la Pi y en la PC.
def _button(parent, text, command, bg=BTN_BG, fg=FG, font=("Arial", -16, "bold"),
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
        # Ventana fija de 800x480 (pantalla de 7'' de la Pi): en la Pi
        # ocupa la pantalla completa (fullscreen, sin barra de titulo); en
        # la PC de desarrollo se ve una ventana identica, centrada. El
        # maxsize/resizable(False) evita que el WM la agrande si algun
        # widget pide mas espacio.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        offset_x = max(0, (screen_w - WINDOW_W) // 2)
        offset_y = max(0, (screen_h - WINDOW_H) // 2)
        root.geometry(f"{WINDOW_W}x{WINDOW_H}+{offset_x}+{offset_y}")
        root.resizable(False, False)
        root.maxsize(WINDOW_W, WINDOW_H)
        if (screen_w, screen_h) == (WINDOW_W, WINDOW_H):
            root.attributes("-fullscreen", True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self._video_w = WINDOW_W - PANEL_WIDTH
        self._video_h = WINDOW_H

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
        top_bar.pack(fill="x", padx=8, pady=(6, 0))

        self._logo_img = self._load_icon("logo.jpg", height=36)
        if self._logo_img is not None:
            tk.Label(top_bar, image=self._logo_img, bg=BG, bd=0).pack(side="left")

        self._gear_img = self._load_icon("config.png", height=28)
        self._btn_config = _button(top_bar, "⚙", self._show_calibration,
                                   font=("Arial", -21), width=3)
        if self._gear_img is not None:
            self._btn_config.config(image=self._gear_img, text="", width=36,
                                    height=36, bg=BG)
        self._btn_config.pack(side="right")

        # Fullscreen sin barra de titulo: el unico camino tactil para
        # cerrar la app es este boton, anclado abajo del panel.
        bottom_row = tk.Frame(panel, bg=BG)
        bottom_row.pack(side="bottom", fill="x", padx=8, pady=(4, 8))
        _button(bottom_row, "Reiniciar", self._on_reset,
                height=2).pack(side="left", expand=True, fill="x", padx=(0, 4))
        _button(bottom_row, "Salir", self._cb.on_close, fg="#ff6b6b",
                height=2).pack(side="left", expand=True, fill="x", padx=(4, 0))

        # --- vista principal ---
        self._main_view = tk.Frame(panel, bg=BG)
        self._main_view.pack(fill="both", expand=True)

        start_stop_row = tk.Frame(self._main_view, bg=BG)
        start_stop_row.pack(fill="x", padx=8, pady=(10, 4))
        self._btn_start = _button(start_stop_row, "Inicio", self._on_start,
                                  bg=ACCENT_START, height=2)
        self._btn_start.config(state="normal" if background_exists else "disabled")
        self._btn_start.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._btn_stop = _button(start_stop_row, "Detener", self._on_stop,
                                 bg=ACCENT_STOP, height=2)
        self._btn_stop.config(state="disabled")
        self._btn_stop.pack(side="left", expand=True, fill="x", padx=(4, 0))

        counts_frame = tk.Frame(self._main_view, bg=BG)
        counts_frame.pack(fill="both", expand=True, padx=10, pady=8)
        count_font = ("Arial", -22, "bold")
        self._fiber_label = tk.Label(counts_frame, text="Fibras: 0", bg=BG,
                                     font=count_font, fg=FIBER_COLOR)
        self._fiber_label.pack(anchor="w", pady=2)
        self._amorf_label = tk.Label(counts_frame, text="Amorfas: 0", bg=BG,
                                     font=count_font, fg=AMORF_COLOR)
        self._amorf_label.pack(anchor="w", pady=2)
        tk.Frame(counts_frame, bg="#333333", height=2).pack(fill="x", pady=4)
        self._total_label = tk.Label(counts_frame, text="Total: 0", bg=BG,
                                     font=("Arial", -28, "bold"), fg=FG)
        self._total_label.pack(anchor="w", pady=2)

        # Segundo divisor: separa los contadores (lo que importa de un
        # vistazo) de la telemetria de la bomba (contexto secundario).
        tk.Frame(counts_frame, bg="#333333", height=2).pack(fill="x", pady=(8, 4))

        pump_font = ("Arial", -16)
        self._caudal_label = tk.Label(counts_frame, text="Caudal: -", bg=BG,
                                      font=pump_font, fg=FG_DIM)
        self._caudal_label.pack(anchor="w", pady=1)
        self._pump_label = tk.Label(counts_frame, text="Bomba: -", bg=BG,
                                    font=pump_font, fg=FG_DIM)
        self._pump_label.pack(anchor="w", pady=1)
        self._membership_label = tk.Label(counts_frame, text="Membresía: -",
                                          bg=BG, font=pump_font, fg=FG_DIM)
        self._membership_label.pack(anchor="w", pady=1)
        self._volume_label = tk.Label(counts_frame, text="Volumen: -", bg=BG,
                                      font=pump_font, fg=FG_DIM)
        self._volume_label.pack(anchor="w", pady=1)

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
        step_font = ("Arial", -17, "bold")

        tk.Label(parent, text="Calibracion", bg=BG, fg=FG,
                 font=("Arial", -20, "bold")).pack(anchor="w", padx=8, pady=(10, 6))

        self._top_value = self._build_crop_row(parent, "Sup", step_font,
                                               self._crop_top, "top")
        self._bottom_value = self._build_crop_row(parent, "Inf", step_font,
                                                  self._crop_bottom, "bottom")

        _button(parent, "Guardar fondo", self._cb.on_capture_background,
                height=2).pack(fill="x", padx=8, pady=(10, 4))

        self._calib_caudal_label = tk.Label(parent, text="Caudal: -", bg=BG,
                                            font=("Arial", -16), fg=FG_DIM)
        self._calib_caudal_label.pack(anchor="w", padx=8, pady=(6, 0))

        _button(parent, "Volver", self._show_main,
                height=2).pack(fill="x", padx=8, pady=(8, 4))

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

    def _on_reset(self) -> None:
        self._cb.on_reset()
        self._fiber_label.config(text="Fibras: 0")
        self._amorf_label.config(text="Amorfas: 0")
        self._total_label.config(text="Total: 0")

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
        self._pump_label.config(text=f"Bomba: {STATE_LABELS[status.state]}",
                                fg=STATE_COLORS[status.state])
        # La telemetria conserva la ultima membresia aunque el fuzzy ya no
        # este activo; fuera de FUZZY_ACTIVO no representa nada actual.
        if status.state == PumpState.FUZZY_ACTIVO:
            membership = MEMBERSHIP_LABELS[status.membership]
            membership_color = MEMBERSHIP_COLORS[status.membership]
        else:
            membership = "-"
            membership_color = FG_DIM
        self._membership_label.config(text=f"Membresía: {membership}",
                                      fg=membership_color)
        self._volume_label.config(text=f"Volumen: {status.volume_ml:.0f} ml")

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
