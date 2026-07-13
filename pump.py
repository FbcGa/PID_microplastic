"""Bomba peristaltica: Arduino esclavo por serial controlando PWM con
logica difusa. Espejo de camera.py: interfaz duck-typed (calibrate(),
start(), stop(), send_count(n), status(), close()) para que main_live.py
y processing.py no sepan si hablan con el Arduino real o un mock.

Protocolo (linea por linea, terminada en \\n):
    Pi -> Arduino: CALIBRATE | START | STOP | N<conteo> | c<caudal>
    Arduino -> Pi: ST=<estado>,PWM=<n>,Q=<caudal>,MEM=<membresia>,VOL=<ml>

El hilo lector de SerialPump corre en background y guarda solo el ultimo
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


class SerialPump:
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
                caudal=float(fields["Q"]),
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


class MockPump:
    """Simula la interfaz de la bomba para desarrollar la UI en la PC sin
    Arduino conectado. No corre fuzzy real: solo un caudal fijo por estado,
    suficiente para ver los labels moverse."""

    def __init__(self):
        self._state = "OFF"
        self._caudal = 0.0

    def calibrate(self) -> None:
        self._state = "LIMPIEZA"
        self._caudal = 300.0

    def start(self) -> None:
        self._state = "FUZZY_ACTIVO"
        self._caudal = 135.0

    def stop(self) -> None:
        self._state = "OFF"
        self._caudal = 0.0

    def send_count(self, count: int) -> None:
        pass

    def status(self) -> PumpStatus:
        membership = "MEDIA" if self._state == "FUZZY_ACTIVO" else "-"
        return PumpStatus(state=self._state, pwm=0, caudal=self._caudal,
                          membership=membership, volume_ml=0.0)

    def close(self) -> None:
        pass


def open_pump(port: str | None):
    """port=None o 'mock' -> MockPump; ruta de dispositivo -> SerialPump.
    Si abrir el puerto falla (Arduino no conectado), cae a MockPump con un
    warning: el conteo debe seguir funcionando aunque la bomba no este."""
    if port is None or port == "mock":
        return MockPump()
    try:
        return SerialPump(port)
    except (serial.SerialException, OSError) as exc:
        logger.warning("No se pudo abrir la bomba en %s (%s); usando MockPump.",
                       port, exc)
        return MockPump()
