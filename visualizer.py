"""Visual annotation of detection results over the frame."""

import cv2
import numpy as np

from detector import DetectionResult, ParticleClass
from tracker import Track, Tracker

_COLORS: dict[ParticleClass, tuple[int, int, int]] = {
    ParticleClass.FIBER: (0, 165, 255), #color orange
    ParticleClass.AMORPHOUS: (0, 255, 0), #color green
    ParticleClass.BUBBLE: (100, 100, 255), #color blue
}

class Visualizer:
    """Draws contours, labels, and the count panel. Never modifies the
    original frame: always works on a copy."""

    def annotate(self, frame: np.ndarray, result: DetectionResult,
                 tracks: list[Track] | None = None,
                 tracker: Tracker | None = None) -> np.ndarray:
        output = frame.copy()

        for detection in result.particles:
            color = _COLORS[detection.particle_class]
            x, y, _, _ = detection.bbox
            cv2.drawContours(output, [detection.contour], -1, color, 2)
            cv2.putText(output, detection.particle_class.value, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if tracks:
            self._draw_ids(output, tracks)

        self._count_panel(output, result, tracker)
        return output

    @staticmethod
    def _draw_ids(image: np.ndarray, tracks: list[Track]) -> None:
        for track in tracks:
            center = (int(track.position[0]), int(track.position[1]))
            color = _COLORS[track.particle_class]
            cv2.putText(image, f"#{track.id}", center,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    @staticmethod
    def _count_panel(image: np.ndarray, result: DetectionResult,
                     tracker: Tracker | None) -> None:
        panel_height = 145 if tracker is not None else 105
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (240, panel_height), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        fibers = result.count(ParticleClass.FIBER)
        amorphous = result.count(ParticleClass.AMORPHOUS)
        cv2.putText(image, f"Fibers:    {fibers}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    _COLORS[ParticleClass.FIBER], 2, cv2.LINE_AA)
        cv2.putText(image, f"Amorphous: {amorphous}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    _COLORS[ParticleClass.AMORPHOUS], 2, cv2.LINE_AA)
        cv2.putText(image, f"Bubbles discarded: {result.discarded_bubbles}",
                    (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    _COLORS[ParticleClass.BUBBLE], 1, cv2.LINE_AA)

        if tracker is not None:
            cv2.putText(image, f"Total fibers:    {tracker.total_fibers}",
                        (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _COLORS[ParticleClass.FIBER], 1, cv2.LINE_AA)
            cv2.putText(image, f"Total amorphous: {tracker.total_amorphous}",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _COLORS[ParticleClass.AMORPHOUS], 1, cv2.LINE_AA)
