"""Drawing/overlay helpers for the live detection+tracking window."""

import cv2

from detector import CLASS_COLORS, CLASS_LABELS, ClassifiedParticle
from tracker import Track, Tracker

FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(frame_bgr, particles: list[ClassifiedParticle],
             tracks: list[Track], tracker: Tracker):
    """Draws each particle's contour by class and each confirmed track's ID."""
    out = frame_bgr.copy()
    for p in particles:
        color = CLASS_COLORS.get(p.label, (200, 200, 200))
        cv2.drawContours(out, [p.contour], -1, color, 2)
    for t in tracks:
        x, y = int(t.position[0]), int(t.position[1])
        color = CLASS_COLORS.get(t.label, (200, 200, 200))
        cv2.circle(out, (x, y), 3, color, -1)
        cv2.putText(out, f"#{t.id} {CLASS_LABELS.get(t.label, t.label)}",
                    (x + 6, y - 6), FONT, 0.5, color, 1)
    _draw_counts(out, len(particles), tracker)
    return out


def _draw_counts(frame_bgr, n_particles: int, tracker: Tracker) -> None:
    """Semi-transparent panel with running unique-particle totals."""
    lines = [
        f"Frame: {n_particles} particulas",
        f"Total fibras: {tracker.total_fibers}",
        f"Total amorfas: {tracker.total_amorphous}",
    ]
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (260, 20 + 24 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
    for i, text in enumerate(lines):
        cv2.putText(frame_bgr, text, (10, 28 + 24 * i), FONT, 0.6,
                    (255, 255, 255), 1)
