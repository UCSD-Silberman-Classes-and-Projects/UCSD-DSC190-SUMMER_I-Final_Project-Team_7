"""
OpenCVConeDetector - classical HSV/contour orange-cone detector.

Implements the same `detect(cam_img, depth_array) -> (raw_detections,
selected)` interface as MockDetector (donkeycar/parts/object_detector/
cone_detector.py) - ConeDetector, ObstaclePlanner, and PilotArbiter don't
know or care which backend produced a Detection. This is the first real
(non-mock) backend, used ahead of a trained model: HSV threshold ->
morphological cleanup -> contour extraction -> area/aspect-ratio filtering
-> bounding box, the same style of pipeline lane_follower.py already uses
for yellow/white line detection, just for orange and reporting boxes
instead of a single scan-row centroid.

No spatial calibration/letterboxing concerns here (unlike a resized-input
model): this operates directly on cam_img at its native IMAGE_W x IMAGE_H,
so bounding boxes are already in the reference frame ObstaclePlanner
expects - see obstacle_types.py's coordinate-convention notes.
"""

import cv2
import numpy as np

from donkeycar.parts.obstacle_types import BBox, Detection


class OpenCVConeDetector:

    def __init__(self, cfg):
        self.hsv_low = np.asarray(getattr(cfg, 'ORANGE_HSV_THRESHOLD_LOW', (5, 120, 120)))
        self.hsv_high = np.asarray(getattr(cfg, 'ORANGE_HSV_THRESHOLD_HIGH', (18, 255, 255)))
        self.morph_kernel_size = getattr(cfg, 'CONE_MORPH_KERNEL_SIZE', 5)
        self.min_area_px = getattr(cfg, 'CONE_MIN_CONTOUR_AREA_PX', 150)
        self.max_area_px = getattr(cfg, 'CONE_MAX_CONTOUR_AREA_PX', 60000)
        # A cone's silhouette is taller than it is wide - rejects wide/flat
        # orange patches (a shirt, reflected glare, a floor patch) that
        # pass the color threshold but aren't cone-shaped. height/width.
        self.min_aspect_ratio = getattr(cfg, 'CONE_MIN_ASPECT_RATIO', 0.8)

    def detect(self, cam_img, depth_array):
        """
        input: cam_img, an RGB numpy array (cfg.IMAGE_W x IMAGE_H); depth_array,
               unused here (distance estimation happens generically in
               ConeDetector via estimate_distance_mm, regardless of backend)
        output: (raw_detections: list[Detection], selected: Detection|None) -
                selected is the largest surviving contour by area, or None
                if nothing passed every filter.
        """
        if cam_img is None:
            return [], None

        hsv = cv2.cvtColor(cam_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_px or area > self.max_area_px:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            if (h / w) < self.min_aspect_ratio:
                continue

            # Classical CV has no real probabilistic score - anything that
            # survives every filter gets a fixed confidence of 1.0 (an
            # all-or-nothing pass, not a true probability). This makes
            # CONE_DETECTOR_MIN_CONFIDENCE's existing filter in ConeDetector
            # effectively a no-op for this backend unless deliberately
            # raised above 1.0-equivalent, i.e. never, so document rather
            # than silently rely on it here.
            detections.append(Detection(
                class_name='traffic_cone',
                confidence=1.0,
                bbox=BBox(float(x), float(y), float(x + w), float(y + h)),
            ))

        if not detections:
            return [], None

        selected = max(detections, key=lambda d: d.bbox.width * d.bbox.height)
        return detections, selected
