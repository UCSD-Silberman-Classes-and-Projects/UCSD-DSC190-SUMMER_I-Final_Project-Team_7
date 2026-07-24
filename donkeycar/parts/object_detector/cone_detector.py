"""
ConeDetector - perception-only object detection for the obstacle-avoidance
project (see the design doc referenced in donkeycar/parts/obstacle_types.py).

The detector interface, a MockDetector backend so the rest of the pipeline
can be built/tested before a trained model exists, and the depth-based
distance helper. Wired into cv_control.py behind OBSTACLE_AVOIDANCE_MODE
!= 'disabled'; only in "shadow"/"active" does anything (ObstaclePlanner)
actually consume its object/* outputs - in "observe" they're diagnostic
only (object/* + overlay), with zero effect on driving.
"""

import logging
import time

import numpy as np

from donkeycar.parts.obstacle_types import BBox, Detection

logger = logging.getLogger(__name__)


def estimate_distance_mm(depth_array, bbox, h_frac=(0.4, 0.6), v_frac=(0.5, 1.0),
                          percentile=25, min_valid_depth_mm=200, min_valid_pixels=20):
    """
    Estimate distance to a detected object from a bounded region of interest
    within its bounding box, using a robust statistic rather than a raw
    minimum.

    Two deliberate choices, per the design doc:
      - The ROI is a configurable inner slice of the bbox (default: middle
        40-60% horizontally, lower 50-100% vertically), not the full box -
        the full box can include background pixels beside/above the object
        (e.g. sky or track surface around a cone).
      - The statistic is a low percentile over valid pixels, not
        np.min() - a single noisy depth pixel hitting the raw minimum could
        trigger a false emergency stop; a percentile is far less sensitive
        to one outlier while still favoring "the nearest real surface."

    Still approximate even with both of the above: the OAK-D's RGB preview
    and its stereo-derived depth map are not perfectly spatially aligned
    (different sensors/baseline/FOV - see donkeycar/parts/oak_d.py and
    object_avoidance.md's identical caveat for ObjectAvoider). This narrows
    the sampling region, it doesn't solve alignment.

    input: depth_array, an HxW uint16 array in mm (0 = no reading), or None
           if depth isn't available (e.g. camera isn't an OAK-D); bbox, a
           BBox in the same reference frame as depth_array; h_frac/v_frac,
           (lo, hi) fractions of the bbox's width/height defining the ROI;
           percentile, the low percentile used as the distance estimate;
           min_valid_depth_mm, readings nearer than this are treated as
           lens-adjacent noise, not real; min_valid_pixels, the ROI must
           have at least this many valid readings or the result is invalid.
    output: (distance_mm: float|None, distance_valid: bool, valid_pixel_count: int)
    """
    if depth_array is None:
        return None, False, 0

    h, w = depth_array.shape[:2]

    x1 = int(round(bbox.x1 + h_frac[0] * bbox.width))
    x2 = int(round(bbox.x1 + h_frac[1] * bbox.width))
    y1 = int(round(bbox.y1 + v_frac[0] * bbox.height))
    y2 = int(round(bbox.y1 + v_frac[1] * bbox.height))

    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None, False, 0

    roi = depth_array[y1:y2, x1:x2]
    valid = roi[roi >= min_valid_depth_mm]
    valid_count = int(valid.size)

    if valid_count < min_valid_pixels:
        return None, False, valid_count

    distance_mm = float(np.percentile(valid, percentile))
    return distance_mm, True, valid_count


class MockDetector:
    """
    Fake detection backend with the same interface a real model backend
    will have (`detect(cam_img, depth_array) -> (raw_detections, selected)`,
    both already in the cam/image_array reference frame - a real backend is
    responsible for its own letterbox unmapping via
    obstacle_types.unletterbox_bbox before returning here).

    Exists so ConeDetector, and everything built on top of it
    (ObstaclePlanner, PilotArbiter), can be built and unit-tested before a
    cone model has been trained. Controllable two ways:
      - `set_detection()`/`clear_detection()`, called directly (tests, or
        an interactive session) to control what the "next" frame detects.
      - `CONE_DETECTOR_MOCK_SCRIPT` in cfg: {frame_id: entry_or_None}, where
        entry is a dict of set_detection() kwargs - lets a scripted sequence
        (e.g. "cone appears at frame 30, gone by frame 90") play out
        automatically for testing temporal behavior.
    """

    def __init__(self, cfg=None):
        self._frame_id = 0
        self._current = None  # dict of set_detection() kwargs, or None
        self._script = getattr(cfg, 'CONE_DETECTOR_MOCK_SCRIPT', {}) or {}

        static = getattr(cfg, 'CONE_DETECTOR_MOCK_DETECTION', None) if cfg is not None else None
        if static:
            self.set_detection(**static)

    def set_detection(self, class_name, confidence, bbox,
                       distance_mm=None, distance_valid=False, depth_valid_pixel_count=0):
        """bbox is an (x1, y1, x2, y2) tuple in the reference frame."""
        self._current = dict(class_name=class_name, confidence=confidence, bbox=bbox,
                              distance_mm=distance_mm, distance_valid=distance_valid,
                              depth_valid_pixel_count=depth_valid_pixel_count)

    def clear_detection(self):
        self._current = None

    def detect(self, cam_img, depth_array):
        self._frame_id += 1
        if self._frame_id in self._script:
            entry = self._script[self._frame_id]
            if entry is None:
                self.clear_detection()
            else:
                self.set_detection(**entry)

        if self._current is None:
            return [], None

        selected = Detection(
            class_name=self._current['class_name'],
            confidence=self._current['confidence'],
            bbox=BBox(*self._current['bbox']),
            distance_mm=self._current['distance_mm'],
            distance_valid=self._current['distance_valid'],
            depth_valid_pixel_count=self._current['depth_valid_pixel_count'],
        )
        return [selected], selected


class ConeDetector:
    """
    Vehicle part. run(cam_img, depth_array) publishes a fixed-order tuple
    matching the object/* Memory keys it's wired to in cv_control.py:
    class, confidence, bbox, center_x, distance_mm, distance_valid,
    depth_valid_pixel_count, frame_id, timestamp, raw_detections.

    Runs on a configurable cadence (OBJECT_DETECTION_HZ <= DRIVE_LOOP_HZ)
    and caches the result between inference ticks - the cached result keeps
    its original inference timestamp (time.monotonic(), not wall-clock) so
    a consumer's staleness check stays meaningful across cached frames; see
    the design doc for why refreshing it on republish would defeat the
    point. Implemented non-threaded - only add threaded=True in
    cv_control.py if an on-Pi benchmark shows the drive loop needs it.
    """

    def __init__(self, cfg):
        self.mode = getattr(cfg, 'CONE_DETECTOR_MODE', 'mock')
        if self.mode == 'mock':
            self._backend = MockDetector(cfg)
        elif self.mode == 'model':
            raise NotImplementedError(
                "CONE_DETECTOR_MODE='model' has no backend yet - training/exporting "
                "a real cone model is Milestone 6/7 of the obstacle-avoidance plan. "
                "Use CONE_DETECTOR_MODE='mock' until then."
            )
        else:
            raise ValueError(f"Unknown CONE_DETECTOR_MODE: {self.mode!r}")

        drive_hz = getattr(cfg, 'DRIVE_LOOP_HZ', 20)
        detection_hz = getattr(cfg, 'OBJECT_DETECTION_HZ', drive_hz)
        self._ticks_between_inference = max(1, round(drive_hz / detection_hz)) if detection_hz > 0 else 1

        self.min_confidence = getattr(cfg, 'CONE_DETECTOR_MIN_CONFIDENCE', 0.5)

        self.depth_roi_h_frac = getattr(cfg, 'DEPTH_ROI_H_FRAC', (0.4, 0.6))
        self.depth_roi_v_frac = getattr(cfg, 'DEPTH_ROI_V_FRAC', (0.5, 1.0))
        self.depth_robust_percentile = getattr(cfg, 'DEPTH_ROBUST_PERCENTILE', 25)
        self.depth_min_valid_pixels = getattr(cfg, 'DEPTH_MIN_VALID_PIXELS', 20)
        self.depth_min_valid_depth_mm = getattr(cfg, 'OBJECT_DETECTOR_MIN_VALID_DEPTH_MM', 200)

        self._tick_count = 0
        self._frame_id = 0
        # cached "batch" as the raw tuple pieces, rather than a DetectionBatch,
        # since that's what run() needs to keep republishing verbatim
        self._cached_raw = []
        self._cached_selected = None
        self._cached_timestamp = None

    def _run_inference(self, cam_img, depth_array):
        self._frame_id += 1
        now = time.monotonic()
        raw, selected = self._backend.detect(cam_img, depth_array)

        raw = [d for d in raw if d.confidence >= self.min_confidence]
        if selected is not None and selected.confidence < self.min_confidence:
            selected = None

        if selected is not None:
            distance_mm, distance_valid, valid_px = estimate_distance_mm(
                depth_array, selected.bbox,
                h_frac=self.depth_roi_h_frac, v_frac=self.depth_roi_v_frac,
                percentile=self.depth_robust_percentile,
                min_valid_depth_mm=self.depth_min_valid_depth_mm,
                min_valid_pixels=self.depth_min_valid_pixels,
            )
            selected.distance_mm = distance_mm
            selected.distance_valid = distance_valid
            selected.depth_valid_pixel_count = valid_px

        self._cached_raw = raw
        self._cached_selected = selected
        self._cached_timestamp = now

    def run(self, cam_img, depth_array):
        self._tick_count += 1
        if cam_img is not None and (self._tick_count % self._ticks_between_inference) == 0:
            self._run_inference(cam_img, depth_array)

        sel = self._cached_selected
        raw_tuples = [
            (d.class_name, d.confidence, (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2),
             d.distance_mm, d.distance_valid, d.depth_valid_pixel_count)
            for d in self._cached_raw
        ]

        if sel is None:
            return (None, None, None, None, None, False, 0,
                    self._frame_id, self._cached_timestamp, raw_tuples)

        bbox_tuple = (sel.bbox.x1, sel.bbox.y1, sel.bbox.x2, sel.bbox.y2)
        return (sel.class_name, sel.confidence, bbox_tuple, sel.center_x,
                sel.distance_mm, sel.distance_valid, sel.depth_valid_pixel_count,
                self._frame_id, self._cached_timestamp, raw_tuples)

    def shutdown(self):
        pass
