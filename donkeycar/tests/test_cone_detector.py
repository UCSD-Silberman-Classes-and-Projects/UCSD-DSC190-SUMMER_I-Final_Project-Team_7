import types
import unittest

import numpy as np

from donkeycar.parts.obstacle_types import BBox
from donkeycar.parts.object_detector.cone_detector import (
    ConeDetector,
    MockDetector,
    estimate_distance_mm,
)


def make_cfg(**overrides):
    cfg = types.SimpleNamespace(
        CONE_DETECTOR_MODE='mock',
        DRIVE_LOOP_HZ=20,
        OBJECT_DETECTION_HZ=20,
        CONE_DETECTOR_MIN_CONFIDENCE=0.5,
        DEPTH_ROI_H_FRAC=(0.4, 0.6),
        DEPTH_ROI_V_FRAC=(0.5, 1.0),
        DEPTH_ROBUST_PERCENTILE=25,
        DEPTH_MIN_VALID_PIXELS=20,
        OBJECT_DETECTOR_MIN_VALID_DEPTH_MM=200,
        CONE_DETECTOR_MOCK_DETECTION=None,
        CONE_DETECTOR_MOCK_SCRIPT=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestEstimateDistanceMm(unittest.TestCase):

    def test_none_depth_array_is_invalid(self):
        distance, valid, count = estimate_distance_mm(None, BBox(0, 0, 10, 10))
        self.assertIsNone(distance)
        self.assertFalse(valid)
        self.assertEqual(0, count)

    def test_samples_only_the_configured_roi_not_the_full_box(self):
        # 100x100 depth image: background at 5000mm everywhere, a 20x20
        # patch at 800mm sitting where the ROI (middle 40-60% h, lower
        # 50-100% v of a 0,0-100,100 bbox) should land.
        depth = np.full((100, 100), 5000, dtype=np.uint16)
        depth[50:100, 40:60] = 800  # ROI region for bbox (0,0,100,100)
        bbox = BBox(0, 0, 100, 100)

        distance, valid, count = estimate_distance_mm(
            depth, bbox, h_frac=(0.4, 0.6), v_frac=(0.5, 1.0),
            percentile=50, min_valid_depth_mm=200, min_valid_pixels=20)

        self.assertTrue(valid)
        self.assertAlmostEqual(800, distance, delta=1)
        self.assertEqual(20 * 50, count)

    def test_below_min_valid_pixels_is_invalid(self):
        depth = np.zeros((100, 100), dtype=np.uint16)  # all zero = all invalid (lens noise)
        bbox = BBox(0, 0, 100, 100)
        distance, valid, count = estimate_distance_mm(depth, bbox, min_valid_pixels=20)
        self.assertIsNone(distance)
        self.assertFalse(valid)
        self.assertEqual(0, count)

    def test_percentile_is_robust_to_a_single_noisy_outlier(self):
        # Real surface at 1000mm; one lone pixel reads an implausibly close
        # 210mm (just above the noise floor) due to sensor noise. A raw
        # min() would report 210mm and could trigger a false emergency
        # stop; the percentile should stay close to the real 1000mm.
        depth = np.full((100, 100), 1000, dtype=np.uint16)
        depth[75, 45] = 210  # one noisy pixel inside the default ROI
        bbox = BBox(0, 0, 100, 100)

        distance, valid, count = estimate_distance_mm(
            depth, bbox, h_frac=(0.4, 0.6), v_frac=(0.5, 1.0),
            percentile=25, min_valid_depth_mm=200, min_valid_pixels=20)

        self.assertTrue(valid)
        self.assertGreater(distance, 900)  # nowhere near the 210mm outlier
        self.assertLess(np.min(depth[50:100, 40:60]), distance)  # raw min is far below the estimate

    def test_roi_clipped_to_array_bounds(self):
        depth = np.full((50, 50), 700, dtype=np.uint16)
        # bbox extends well past the array on every side - ROI computation
        # must clip to the array, not crash or wrap. Use the full bbox as
        # the ROI (h_frac/v_frac = 0..1) to isolate the clamping behavior
        # itself from the default inner-ROI fraction narrowing things further.
        bbox = BBox(-20, -20, 200, 200)
        distance, valid, count = estimate_distance_mm(
            depth, bbox, h_frac=(0.0, 1.0), v_frac=(0.0, 1.0), min_valid_pixels=1)
        self.assertTrue(valid)
        self.assertAlmostEqual(700, distance, delta=1)
        self.assertEqual(50 * 50, count)

    def test_roi_entirely_outside_array_is_invalid_not_a_crash(self):
        depth = np.full((50, 50), 700, dtype=np.uint16)
        # bbox is huge, and the default inner ROI fraction (middle 40-60%
        # horizontally, lower half vertically) lands entirely past the
        # array's right/bottom edge - must return invalid cleanly.
        bbox = BBox(-20, -20, 200, 200)
        distance, valid, count = estimate_distance_mm(depth, bbox, min_valid_pixels=1)
        self.assertIsNone(distance)
        self.assertFalse(valid)
        self.assertEqual(0, count)


class TestMockDetector(unittest.TestCase):

    def test_no_detection_by_default(self):
        mock = MockDetector()
        raw, selected = mock.detect(cam_img=None, depth_array=None)
        self.assertEqual([], raw)
        self.assertIsNone(selected)

    def test_set_and_clear_detection(self):
        mock = MockDetector()
        mock.set_detection('traffic_cone', 0.9, (10, 10, 30, 40))
        raw, selected = mock.detect(None, None)
        self.assertEqual(1, len(raw))
        self.assertEqual('traffic_cone', selected.class_name)
        self.assertEqual(0.9, selected.confidence)

        mock.clear_detection()
        raw, selected = mock.detect(None, None)
        self.assertEqual([], raw)
        self.assertIsNone(selected)

    def test_static_detection_from_cfg(self):
        cfg = make_cfg(CONE_DETECTOR_MOCK_DETECTION=dict(
            class_name='traffic_cone', confidence=0.8, bbox=(0, 0, 10, 10)))
        mock = MockDetector(cfg)
        _, selected = mock.detect(None, None)
        self.assertIsNotNone(selected)
        self.assertEqual('traffic_cone', selected.class_name)

    def test_script_drives_detection_over_frames(self):
        cfg = make_cfg(CONE_DETECTOR_MOCK_SCRIPT={
            2: dict(class_name='traffic_cone', confidence=0.9, bbox=(0, 0, 10, 10)),
            4: None,
        })
        mock = MockDetector(cfg)

        _, sel1 = mock.detect(None, None)  # frame 1: nothing scripted yet
        self.assertIsNone(sel1)
        _, sel2 = mock.detect(None, None)  # frame 2: scripted detection appears
        self.assertIsNotNone(sel2)
        _, sel3 = mock.detect(None, None)  # frame 3: still holds from frame 2
        self.assertIsNotNone(sel3)
        _, sel4 = mock.detect(None, None)  # frame 4: scripted clear
        self.assertIsNone(sel4)


class TestConeDetector(unittest.TestCase):

    def test_unknown_mode_raises(self):
        cfg = make_cfg(CONE_DETECTOR_MODE='bogus')
        with self.assertRaises(ValueError):
            ConeDetector(cfg)

    def test_model_mode_not_implemented_yet(self):
        cfg = make_cfg(CONE_DETECTOR_MODE='model')
        with self.assertRaises(NotImplementedError):
            ConeDetector(cfg)

    def test_output_tuple_shape_when_nothing_detected(self):
        cfg = make_cfg()
        detector = ConeDetector(cfg)
        cam_img = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detector.run(cam_img, None)
        self.assertEqual(10, len(result))
        (class_name, confidence, bbox, center_x, distance_mm, distance_valid,
         depth_valid_pixel_count, frame_id, timestamp, raw) = result
        self.assertIsNone(class_name)
        self.assertIsNone(confidence)
        self.assertIsNone(bbox)
        self.assertIsNone(center_x)
        self.assertIsNone(distance_mm)
        self.assertFalse(distance_valid)
        self.assertEqual(0, depth_valid_pixel_count)
        self.assertEqual([], raw)

    def test_detection_passes_through_with_center_x_and_distance(self):
        cfg = make_cfg(CONE_DETECTOR_MOCK_DETECTION=dict(
            class_name='traffic_cone', confidence=0.9, bbox=(100, 100, 140, 160)))
        detector = ConeDetector(cfg)
        cam_img = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 900, dtype=np.uint16)

        result = detector.run(cam_img, depth)
        (class_name, confidence, bbox, center_x, distance_mm, distance_valid,
         depth_valid_pixel_count, frame_id, timestamp, raw) = result

        self.assertEqual('traffic_cone', class_name)
        self.assertEqual(0.9, confidence)
        self.assertEqual((100, 100, 140, 160), bbox)
        self.assertEqual(120, center_x)
        self.assertTrue(distance_valid)
        self.assertAlmostEqual(900, distance_mm, delta=1)
        self.assertGreater(depth_valid_pixel_count, 0)
        self.assertEqual(1, frame_id)
        self.assertIsNotNone(timestamp)

    def test_low_confidence_detection_is_filtered_out(self):
        cfg = make_cfg(CONE_DETECTOR_MIN_CONFIDENCE=0.5,
                        CONE_DETECTOR_MOCK_DETECTION=dict(
                            class_name='traffic_cone', confidence=0.2, bbox=(0, 0, 10, 10)))
        detector = ConeDetector(cfg)
        cam_img = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detector.run(cam_img, None)
        self.assertIsNone(result[0])  # class_name
        self.assertEqual([], result[9])  # raw_detections also filtered

    def test_inference_only_runs_at_configured_cadence(self):
        # DRIVE_LOOP_HZ=20, OBJECT_DETECTION_HZ=5 -> infer every 4th tick
        cfg = make_cfg(DRIVE_LOOP_HZ=20, OBJECT_DETECTION_HZ=5,
                        CONE_DETECTOR_MOCK_DETECTION=dict(
                            class_name='traffic_cone', confidence=0.9, bbox=(0, 0, 10, 10)))
        detector = ConeDetector(cfg)
        cam_img = np.zeros((240, 320, 3), dtype=np.uint8)

        frame_ids = [detector.run(cam_img, None)[7] for _ in range(8)]
        # frame_id (inference counter) should only increment on ticks 4 and 8
        self.assertEqual([0, 0, 0, 1, 1, 1, 1, 2], frame_ids)

    def test_cached_timestamp_does_not_change_between_inference_ticks(self):
        cfg = make_cfg(DRIVE_LOOP_HZ=20, OBJECT_DETECTION_HZ=5,
                        CONE_DETECTOR_MOCK_DETECTION=dict(
                            class_name='traffic_cone', confidence=0.9, bbox=(0, 0, 10, 10)))
        detector = ConeDetector(cfg)
        cam_img = np.zeros((240, 320, 3), dtype=np.uint8)

        timestamps = [detector.run(cam_img, None)[8] for _ in range(4)]
        # ticks 1-3 republish the same cached (None, since inference hasn't
        # run yet) timestamp; tick 4 is the first real inference
        self.assertIsNone(timestamps[0])
        self.assertIsNone(timestamps[1])
        self.assertIsNone(timestamps[2])
        self.assertIsNotNone(timestamps[3])

    def test_none_cam_img_does_not_crash_and_skips_inference(self):
        cfg = make_cfg(CONE_DETECTOR_MOCK_DETECTION=dict(
            class_name='traffic_cone', confidence=0.9, bbox=(0, 0, 10, 10)))
        detector = ConeDetector(cfg)
        result = detector.run(None, None)
        self.assertIsNone(result[0])
        self.assertEqual(0, result[7])  # frame_id never advanced


if __name__ == '__main__':
    unittest.main()
