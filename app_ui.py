"""Tkinter touch UI: live video + config panel. No CV logic here — it only
renders FrameResult snapshots pulled off the worker's queue and forwards
button presses to the callbacks it's given (wired up in main_live.py).
"""

import queue
import tkinter as tk
from dataclasses import dataclass
from typing import Callable

import cv2
from PIL import Image, ImageTk

from processing import FrameResult, Mode

POLL_MS = 33
CROP_STEP = 5


@dataclass
class Callbacks:
    on_calibrate: Callable[[], None]
    on_crop_change: Callable[[int, int], None]
    on_capture_background: Callable[[], None]
    on_save_calibration: Callable[[], None]
    on_start: Callable[[], None]
    on_stop: Callable[[], None]


class App:
    def __init__(self, root: tk.Tk, results: "queue.Queue[FrameResult]",
                 callbacks: Callbacks, initial_crop_top: int,
                 initial_crop_bottom: int, background_exists: bool):
        self._root = root
        self._queue = results
        self._cb = callbacks
        self._crop_top = initial_crop_top
        self._crop_bottom = initial_crop_bottom
        self._photo: ImageTk.PhotoImage | None = None  # referencia viva, Tkinter la descarta si no

        root.title("Microplasticos - captura en vivo")
        root.attributes("-fullscreen", True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self._video_label = tk.Label(root, bg="black")
        self._video_label.grid(row=0, column=0, sticky="nsew")

        panel = tk.Frame(root, width=260)
        panel.grid(row=0, column=1, sticky="ns")

        btn_font = ("Arial", 16, "bold")
        self._btn_calibrate = tk.Button(panel, text="Calibracion", font=btn_font,
                                        height=2, command=self._on_calibrate)
        self._btn_calibrate.pack(fill="x", padx=8, pady=8)

        self._btn_start = tk.Button(
            panel, text="Inicio", font=btn_font, height=2, command=self._on_start,
            state="normal" if background_exists else "disabled")
        self._btn_start.pack(fill="x", padx=8, pady=8)

        self._btn_stop = tk.Button(panel, text="Detener", font=btn_font, height=2,
                                   command=self._on_stop, state="disabled")
        self._btn_stop.pack(fill="x", padx=8, pady=8)

        counts_frame = tk.Frame(panel)
        counts_frame.pack(fill="x", padx=8, pady=16)
        self._fiber_label = tk.Label(counts_frame, text="Fibras: 0", font=("Arial", 14))
        self._fiber_label.pack(anchor="w")
        self._amorf_label = tk.Label(counts_frame, text="Amorfas: 0", font=("Arial", 14))
        self._amorf_label.pack(anchor="w")

        self._calib_panel = tk.Frame(panel)
        self._build_calibration_panel(self._calib_panel, btn_font)

        root.after(POLL_MS, self._poll_queue)

    def set_start_enabled(self, enabled: bool) -> None:
        self._btn_start.config(state="normal" if enabled else "disabled")

    def _build_calibration_panel(self, parent: tk.Frame, btn_font: tuple) -> None:
        step_font = ("Arial", 14, "bold")

        top_row = tk.Frame(parent)
        top_row.pack(fill="x", pady=4)
        tk.Label(top_row, text="Sup", font=step_font).pack(side="left")
        tk.Button(top_row, text="-", font=step_font, width=3,
                  command=lambda: self._adjust_crop(top_delta=-CROP_STEP)).pack(side="left")
        self._top_value = tk.Label(top_row, text=str(self._crop_top), font=step_font, width=4)
        self._top_value.pack(side="left")
        tk.Button(top_row, text="+", font=step_font, width=3,
                  command=lambda: self._adjust_crop(top_delta=CROP_STEP)).pack(side="left")

        bottom_row = tk.Frame(parent)
        bottom_row.pack(fill="x", pady=4)
        tk.Label(bottom_row, text="Inf", font=step_font).pack(side="left")
        tk.Button(bottom_row, text="-", font=step_font, width=3,
                  command=lambda: self._adjust_crop(bottom_delta=-CROP_STEP)).pack(side="left")
        self._bottom_value = tk.Label(bottom_row, text=str(self._crop_bottom), font=step_font, width=4)
        self._bottom_value.pack(side="left")
        tk.Button(bottom_row, text="+", font=step_font, width=3,
                  command=lambda: self._adjust_crop(bottom_delta=CROP_STEP)).pack(side="left")

        tk.Button(parent, text="Capturar fondo", font=btn_font,
                  command=self._cb.on_capture_background).pack(fill="x", pady=4)
        tk.Button(parent, text="Guardar y salir", font=btn_font,
                  command=self._on_save_calibration).pack(fill="x", pady=4)

    def _adjust_crop(self, top_delta: int = 0, bottom_delta: int = 0) -> None:
        self._crop_top = max(0, self._crop_top + top_delta)
        self._crop_bottom = max(0, self._crop_bottom + bottom_delta)
        self._top_value.config(text=str(self._crop_top))
        self._bottom_value.config(text=str(self._crop_bottom))
        self._cb.on_crop_change(self._crop_top, self._crop_bottom)

    def _on_calibrate(self) -> None:
        self._calib_panel.pack(fill="x", padx=8, pady=8)
        self._cb.on_calibrate()

    def _on_save_calibration(self) -> None:
        self._calib_panel.pack_forget()
        self._cb.on_save_calibration()

    def _on_start(self) -> None:
        self._cb.on_start()
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._btn_calibrate.config(state="disabled")

    def _on_stop(self) -> None:
        self._cb.on_stop()
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._btn_calibrate.config(state="normal")

    def _poll_queue(self) -> None:
        try:
            result = self._queue.get_nowait()
        except queue.Empty:
            pass
        else:
            self._render(result)
        self._root.after(POLL_MS, self._poll_queue)

    def _render(self, result: FrameResult) -> None:
        frame_rgb = cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        label_w = self._video_label.winfo_width()
        label_h = self._video_label.winfo_height()
        if label_w > 1 and label_h > 1:
            scale = min(label_w / image.width, label_h / image.height)
            new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(new_size)

        self._photo = ImageTk.PhotoImage(image)
        self._video_label.configure(image=self._photo)

        if result.mode == Mode.RUNNING:
            self._fiber_label.config(text=f"Fibras: {result.fibers}")
            self._amorf_label.config(text=f"Amorfas: {result.amorphous}")
