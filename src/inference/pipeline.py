"""Video and image inference pipeline."""

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False

from src.models.detector import BaseDetector, DetectionResult
from src.evaluation.visualization import draw_boxes


class InferencePipeline:
    """Run a detector on video files or image directories.

    Args:
        detector: An object implementing :class:`BaseDetector`.
        score_threshold: Minimum confidence to keep detections.
        class_names: Optional mapping from label id to name.
    """

    def __init__(
        self,
        detector: BaseDetector,
        score_threshold: float = 0.25,
        class_names: Optional[dict] = None,
    ) -> None:
        self.detector = detector
        self.score_threshold = score_threshold
        self.class_names = class_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_on_image(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a single image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB array.

        Returns:
            :class:`DetectionResult`.
        """
        return self.detector.predict(
            image, score_threshold=self.score_threshold
        )

    def run_on_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
    ) -> List[DetectionResult]:
        """Run detection on every image in a directory.

        Args:
            input_dir: Path to directory of images.
            output_dir: If provided, save annotated images here.

        Returns:
            List of :class:`DetectionResult` objects.
        """
        if not HAS_CV2:
            raise ImportError("opencv-python is required for inference")

        input_path = Path(input_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        image_files = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in exts
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results: List[DetectionResult] = []
        for img_file in image_files:
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = self.run_on_image(img_rgb)
            results.append(result)

            if output_dir:
                vis = draw_boxes(
                    img_rgb, result.boxes, result.labels,
                    result.scores, class_names=self.class_names,
                )
                out_path = os.path.join(output_dir, img_file.name)
                cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        return results

    def run_on_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
    ) -> List[DetectionResult]:
        """Run detection on a video file.

        Args:
            video_path: Path to input video.
            output_path: If provided, write annotated video here.

        Returns:
            List of per-frame :class:`DetectionResult` objects.
        """
        if not HAS_CV2:
            raise ImportError("opencv-python is required for video inference")

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        results: List[DetectionResult] = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            result = self.run_on_image(frame_rgb)
            elapsed = time.time() - t0
            results.append(result)

            frame_count += 1

            if writer is not None:
                vis = draw_boxes(
                    frame_rgb, result.boxes, result.labels,
                    result.scores, class_names=self.class_names,
                )
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                fps_text = f"FPS: {1.0 / max(elapsed, 1e-6):.1f}"
                cv2.putText(
                    vis_bgr, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                )
                writer.write(vis_bgr)

        cap.release()
        if writer is not None:
            writer.release()

        return results
