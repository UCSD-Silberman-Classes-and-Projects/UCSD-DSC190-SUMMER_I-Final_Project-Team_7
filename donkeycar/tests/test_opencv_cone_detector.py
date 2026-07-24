import types
import unittest

import cv2
import numpy as np

from donkeycar.parts.object_detector.opencv_cone_detector import OpenCVConeDetector


def make_cfg(**overrides):
    cfg = types.SimpleNamespace(
        ORANGE_HSV_THRESHOLD_LOW=(5, 120, 120),
        ORANGE_HSV_THRESHOLD_HIGH=(18, 255, 255),
        CONE_MORPH_KERNEL_SIZE=5,
        CONE_MIN_CONTOUR_AREA_PX=150,
        CONE_MAX_CONTOUR_AREA_PX=60000,
        CONE_MIN_ASPECT_RATIO=0.8,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def hsv_to_rgb(h, s, v):
    """A single HSV pixel, converted to the RGB tuple that (after the
    detector's own RGB->HSV conversion) lands back at (h, s, v) - used to
    draw synthetic test shapes in a color guaranteed to fall inside the
    detector's threshold range, rather than guessing an RGB triplet."""
    pixel = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


ORANGE_RGB = hsv_to_rgb(10, 200, 200)   # inside the default threshold range
GRAY_RGB = (60, 60, 60)                  # low saturation - well outside orange range


class TestOpenCVConeDetector(unittest.TestCase):

    def _blank_image(self, color=GRAY_RGB):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:, :] = color
        return img

    def test_none_image_returns_nothing(self):
        detector = OpenCVConeDetector(make_cfg())
        raw, selected = detector.detect(None, None)
        self.assertEqual([], raw)
        self.assertIsNone(selected)

    def test_no_orange_returns_nothing(self):
        detector = OpenCVConeDetector(make_cfg())
        img = self._blank_image()
        raw, selected = detector.detect(img, None)
        self.assertEqual([], raw)
        self.assertIsNone(selected)

    def test_detects_a_cone_shaped_orange_blob(self):
        detector = OpenCVConeDetector(make_cfg())
        img = self._blank_image()
        # A tall rectangle (width 30, height 60 -> aspect ratio 2.0, area
        # 1800) - well inside all the default filters.
        cv2.rectangle(img, (100, 100), (130, 160), color=ORANGE_RGB, thickness=-1)

        raw, selected = detector.detect(img, None)

        self.assertEqual(1, len(raw))
        self.assertIsNotNone(selected)
        self.assertEqual('traffic_cone', selected.class_name)
        self.assertEqual(1.0, selected.confidence)
        # bounding box should tightly match the drawn rectangle (allow a
        # couple pixels of slack for anti-aliasing/contour rounding)
        self.assertAlmostEqual(100, selected.bbox.x1, delta=3)
        self.assertAlmostEqual(100, selected.bbox.y1, delta=3)
        self.assertAlmostEqual(130, selected.bbox.x2, delta=3)
        self.assertAlmostEqual(160, selected.bbox.y2, delta=3)

    def test_rejects_blob_smaller_than_min_area(self):
        detector = OpenCVConeDetector(make_cfg(CONE_MIN_CONTOUR_AREA_PX=150))
        img = self._blank_image()
        # 5x5 = 25px^2, well below the 150px^2 floor
        cv2.rectangle(img, (100, 100), (105, 105), color=ORANGE_RGB, thickness=-1)

        raw, selected = detector.detect(img, None)
        self.assertEqual([], raw)
        self.assertIsNone(selected)

    def test_rejects_wide_flat_blob_via_aspect_ratio(self):
        detector = OpenCVConeDetector(make_cfg(CONE_MIN_ASPECT_RATIO=0.8))
        img = self._blank_image()
        # width 150, height 20 -> aspect ratio ~0.13, area 3000 (plenty of
        # area, but the wrong shape for a cone - e.g. glare off the floor)
        cv2.rectangle(img, (50, 100), (200, 120), color=ORANGE_RGB, thickness=-1)

        raw, selected = detector.detect(img, None)
        self.assertEqual([], raw)
        self.assertIsNone(selected)

    def test_rejects_blob_larger_than_max_area(self):
        detector = OpenCVConeDetector(make_cfg(CONE_MAX_CONTOUR_AREA_PX=500))
        img = self._blank_image()
        cv2.rectangle(img, (50, 50), (110, 110), color=ORANGE_RGB, thickness=-1)  # 60x60=3600px^2

        raw, selected = detector.detect(img, None)
        self.assertEqual([], raw)
        self.assertIsNone(selected)

    def test_selected_is_the_largest_of_multiple_detections(self):
        detector = OpenCVConeDetector(make_cfg())
        img = self._blank_image()
        cv2.rectangle(img, (20, 20), (40, 60), color=ORANGE_RGB, thickness=-1)     # smaller: 20x40
        cv2.rectangle(img, (200, 100), (240, 200), color=ORANGE_RGB, thickness=-1)  # larger: 40x100

        raw, selected = detector.detect(img, None)
        self.assertEqual(2, len(raw))
        self.assertAlmostEqual(200, selected.bbox.x1, delta=3)


if __name__ == '__main__':
    unittest.main()
