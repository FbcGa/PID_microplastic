"""Bomba peristaltica: Arduino esclavo por serial controlando PWM con
logica difusa. Toda la comunicacion vive en la clase Arduino; el resto
de la app solo llama sus metodos (calibrate(), start(), stop(),
send_count(n), status(), close()).

Protocolo (linea por linea, terminada en \\n):
    Pi -> Arduino: CALIBRATE | START | STOP | N<conteo> | c<caudal>
    Arduino -> Pi: ST=<estado>,PWM=<n>,CS=<caudal>,MEM=<membresia>,VOL=<ml>

El hilo lector de Arduino corre en background y guarda solo el ultimo
PumpStatus recibido; status() lo lee sin bloquear (lock corto).
"""

import logging
import threading
import time
from dataclasses import dataclass

import serial

logger = logging.getLogger(__name__)

BAUDRATE = 9600
# Abrir el puerto resetea el Arduino (auto-reset via DTR); hay que esperar
# a que el bootloader termine antes de mandar el primer comando.
ARDUINO_RESET_S = 2.0


@dataclass(frozen=True)
class PumpStatus:
    state: str
    pwm: int
    caudal: float
    membership: str
    volume_ml: float


IDLE_STATUS = PumpStatus(state="OFF", pwm=0, caudal=0.0, membership="-", volume_ml=0.0)


class Arduino:
    """Bomba real: manda comandos por serial y parsea la telemetria que el
    Arduino imprime una vez por segundo (linea ST=...)."""

    def __init__(self, port: str, baudrate: int = BAUDRATE):
        self._serial = serial.Serial(port, baudrate, timeout=1)
        self._write_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status = IDLE_STATUS
        self._stop_requested = False
        time.sleep(ARDUINO_RESET_S)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while not self._stop_requested:
            try:
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
            except (serial.SerialException, OSError) as exc:
                if self._stop_requested:  # close() cerro el puerto, no es error
                    break
                logger.warning("Error leyendo de la bomba: %s", exc)
                continue
            if not line:
                continue
            status = self._parse_status(line)
            if status is not None:
                with self._status_lock:
                    self._status = status

    @staticmethod
    def _parse_status(line: str) -> "PumpStatus | None":
        if not line.startswith("ST="):
            return None
        fields = {}
        for part in line.split(","):
            if "=" not in part:
                return None
            key, value = part.split("=", 1)
            fields[key] = value
        try:
            return PumpStatus(
                state=fields["ST"],
                pwm=int(fields["PWM"]),
                caudal=float(fields["CS"]),
                membership=fields["MEM"],
                volume_ml=float(fields["VOL"]),
            )
        except (KeyError, ValueError):
            logger.warning("Linea de telemetria invalida: %s", line)
            return None

    def _send(self, command: str) -> None:
        with self._write_lock:
            try:
                self._serial.write(f"{command}\n".encode("ascii"))
            except (serial.SerialException, OSError) as exc:
                logger.warning("Error mandando '%s' a la bomba: %s", command, exc)

    def calibrate(self) -> None:
        self._send("CALIBRATE")

    def start(self) -> None:
        self._send("START")

    def stop(self) -> None:
        self._send("STOP")

    def send_count(self, count: int) -> None:
        self._send(f"N{count}")

    def status(self) -> PumpStatus:
        with self._status_lock:
            return self._status

    def close(self) -> None:
        self._stop_requested = True
        self._serial.close()


def open_arduino(port: str) -> Arduino:
    """Abre la bomba en el puerto dado. Si el puerto no existe o esta
    ocupado, serial.SerialException sube tal cual: la app debe fallar
    visible en vez de arrancar sin bomba (mismo criterio fail-fast que
    el modelo RF en main_live.py)."""
    return Arduino(port)
