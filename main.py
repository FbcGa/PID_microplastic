"""Entry point: microplastic detection over a video.

Usage:
    uv run main.py video.mp4                  (background = first video frame)
    uv run main.py video.mp4 --background background.jpg

Keys during playback:
    q / ESC  quit
    space    pause / resume
    m        show / hide the binary mask
"""

import argparse

import cv2

from config import DetectionConfig, TrackerConfig
from detector import MicroplasticDetector
from tracker import Tracker
from video_source import FileVideoSource
from visualizer import Visualizer

WINDOW = "Microplastic detection"
MASK_WINDOW = "Mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microplastic detection via background subtraction.")
    parser.add_argument("video", help="Path to the test video")
    parser.add_argument(
        "--background", default=None,
        help="Clean background image; if omitted, the first frame is used")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    detector = MicroplasticDetector(DetectionConfig())
    tracker = Tracker(TrackerConfig())
    visualizer = Visualizer()

    if args.background is not None:
        background = cv2.imread(args.background)
        if background is None:
            raise IOError(f"Could not load background: {args.background}")
        detector.set_background(background)

    paused = False
    show_mask = False
    mask_window_created = False

    with FileVideoSource(args.video) as source:
        print(f"Video: {source.total_frames} frames at {source.fps:.1f} FPS")
        delay_ms = max(1, int(1000 / source.fps)) if source.fps > 0 else 30

        while True:
            if not paused:
                frame = source.read()
                if frame is None:
                    break

                if not detector.has_background:
                    detector.set_background(frame)
                    print("Background taken from the first video frame")

                result = detector.detect(frame)
                tracks = tracker.update(result.particles)
                annotated = visualizer.annotate(frame, result, tracks, tracker)
                cv2.imshow(WINDOW, annotated)
                if show_mask:
                    cv2.imshow(MASK_WINDOW, result.mask)
                    mask_window_created = True

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):  # 27 = ESC
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("m"):
                show_mask = not show_mask
                if not show_mask and mask_window_created:
                    cv2.destroyWindow(MASK_WINDOW)
                    mask_window_created = False

        print("Playback finished.")
        print(f"Total fibers: {tracker.total_fibers}")
        print(f"Total amorphous: {tracker.total_amorphous}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())
