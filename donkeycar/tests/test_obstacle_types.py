import unittest

from donkeycar.parts.obstacle_types import (
    BBox,
    Detection,
    DetectionBatch,
    LaneGeometry,
    detection_batch_from_memory,
    detection_from_values,
    lane_geometry_from_memory,
    unletterbox_bbox,
)


class TestBBox(unittest.TestCase):

    def test_dimensions_and_center(self):
        box = BBox(10, 20, 50, 80)
        self.assertEqual(40, box.width)
        self.assertEqual(60, box.height)
        self.assertEqual(30, box.center_x)
        self.assertEqual(50, box.center_y)

    def test_degenerate_box_has_zero_size_not_negative(self):
        # x2 < x1 shouldn't happen, but width/height must never go negative
        box = BBox(50, 50, 10, 10)
        self.assertEqual(0.0, box.width)
        self.assertEqual(0.0, box.height)

    def test_overlaps_x(self):
        box = BBox(100, 0, 150, 50)
        self.assertTrue(box.overlaps_x(120, 200))   # overlapping on the left edge of the span
        self.assertTrue(box.overlaps_x(50, 120))    # overlapping on the right edge of the span
        self.assertTrue(box.overlaps_x(90, 160))    # span fully contains the box
        self.assertFalse(box.overlaps_x(151, 200))  # entirely to the right
        self.assertFalse(box.overlaps_x(0, 99))     # entirely to the left
        self.assertFalse(box.overlaps_x(150, 200))  # touching, not overlapping


class TestUnletterboxBBox(unittest.TestCase):

    def test_identity_when_model_matches_reference_frame(self):
        box = unletterbox_bbox(10, 20, 50, 80, model_w=320, model_h=240, orig_w=320, orig_h=240)
        self.assertAlmostEqual(10, box.x1)
        self.assertAlmostEqual(20, box.y1)
        self.assertAlmostEqual(50, box.x2)
        self.assertAlmostEqual(80, box.y2)

    def test_vertical_padding_for_wide_reference_frame_in_square_model(self):
        # A 426x240 (landscape) camera frame fit into a 240x240 square
        # model: width is the binding constraint (scale = 240/426), so the
        # scaled image fills the model's width exactly and leaves padding
        # on top/bottom (y), not left/right.
        orig_w, orig_h = 426, 240
        model_w, model_h = 240, 240
        scale = min(model_w / orig_w, model_h / orig_h)
        pad_x = (model_w - orig_w * scale) / 2.0
        pad_y = (model_h - orig_h * scale) / 2.0
        self.assertAlmostEqual(0, pad_x, places=4)
        self.assertGreater(pad_y, 0)

        box = unletterbox_bbox(pad_x, pad_y, pad_x + 10 * scale, pad_y + 10 * scale,
                                model_w, model_h, orig_w, orig_h)
        self.assertAlmostEqual(0, box.x1, places=4)
        self.assertAlmostEqual(0, box.y1, places=4)
        self.assertAlmostEqual(10, box.x2, places=4)
        self.assertAlmostEqual(10, box.y2, places=4)

    def test_horizontal_padding_for_tall_reference_frame_in_square_model(self):
        # A 240x426 (portrait) camera frame fit into a 240x240 square
        # model: height is the binding constraint, so padding lands on
        # left/right (x), not top/bottom.
        orig_w, orig_h = 240, 426
        model_w, model_h = 240, 240
        scale = min(model_w / orig_w, model_h / orig_h)
        pad_x = (model_w - orig_w * scale) / 2.0
        pad_y = (model_h - orig_h * scale) / 2.0
        self.assertGreater(pad_x, 0)
        self.assertAlmostEqual(0, pad_y, places=4)

        box = unletterbox_bbox(pad_x, pad_y, pad_x + 10 * scale, pad_y + 10 * scale,
                                model_w, model_h, orig_w, orig_h)
        self.assertAlmostEqual(0, box.x1, places=4)
        self.assertAlmostEqual(0, box.y1, places=4)
        self.assertAlmostEqual(10, box.x2, places=4)
        self.assertAlmostEqual(10, box.y2, places=4)

    def test_non_square_aspect_ratio_no_padding_needed(self):
        # model input aspect ratio exactly matches the camera frame's -
        # pure uniform scale, no letterbox padding on either axis.
        orig_w, orig_h = 426, 240
        model_w, model_h = 213, 120  # exactly half size, same aspect ratio
        box = unletterbox_bbox(0, 0, 213, 120, model_w, model_h, orig_w, orig_h)
        self.assertAlmostEqual(0, box.x1)
        self.assertAlmostEqual(0, box.y1)
        self.assertAlmostEqual(426, box.x2)
        self.assertAlmostEqual(240, box.y2)

    def test_box_touching_model_input_edge_maps_to_reference_frame_edge(self):
        orig_w, orig_h = 426, 240
        model_w, model_h = 240, 240
        box = unletterbox_bbox(0, 0, model_w, model_h, model_w, model_h, orig_w, orig_h)
        # the full letterboxed frame (including padding) should map back to
        # something that still spans the full original width once unpadded
        self.assertLessEqual(box.x1, 0.5)
        self.assertGreaterEqual(box.x2, orig_w - 0.5)


class TestDetection(unittest.TestCase):

    def test_center_x_delegates_to_bbox(self):
        det = Detection(class_name="traffic_cone", confidence=0.9,
                         bbox=BBox(10, 10, 30, 40))
        self.assertEqual(20, det.center_x)

    def test_defaults_have_no_distance(self):
        det = Detection(class_name="traffic_cone", confidence=0.5, bbox=BBox(0, 0, 1, 1))
        self.assertIsNone(det.distance_mm)
        self.assertFalse(det.distance_valid)
        self.assertEqual(0, det.depth_valid_pixel_count)


class TestDetectionFromValues(unittest.TestCase):

    def test_none_class_name_returns_none(self):
        self.assertIsNone(detection_from_values(None, 0.9, (0, 0, 10, 10)))

    def test_none_bbox_returns_none(self):
        self.assertIsNone(detection_from_values("traffic_cone", 0.9, None))

    def test_builds_detection_with_distance(self):
        det = detection_from_values("traffic_cone", 0.87, (1, 2, 3, 4),
                                     distance_mm=850, distance_valid=True,
                                     depth_valid_pixel_count=42)
        self.assertEqual("traffic_cone", det.class_name)
        self.assertEqual(0.87, det.confidence)
        self.assertEqual(BBox(1, 2, 3, 4), det.bbox)
        self.assertEqual(850, det.distance_mm)
        self.assertTrue(det.distance_valid)
        self.assertEqual(42, det.depth_valid_pixel_count)

    def test_missing_confidence_defaults_to_zero_not_none(self):
        det = detection_from_values("traffic_cone", None, (0, 0, 1, 1))
        self.assertEqual(0.0, det.confidence)


class TestDetectionBatch(unittest.TestCase):

    def test_age_and_staleness(self):
        batch = DetectionBatch(inference_timestamp=100.0)
        self.assertEqual(5.0, batch.age(105.0))
        self.assertFalse(batch.is_stale(105.0, max_age_s=10.0))
        self.assertTrue(batch.is_stale(115.0, max_age_s=10.0))

    def test_missing_timestamp_is_always_infinitely_stale(self):
        batch = DetectionBatch(inference_timestamp=None)
        self.assertEqual(float('inf'), batch.age(0.0))
        self.assertTrue(batch.is_stale(0.0, max_age_s=1e9))

    def test_republishing_a_cached_batch_must_not_refresh_its_timestamp(self):
        # This is the exact bug review round 3 flagged: a cached detection
        # re-published on a later drive-loop tick must keep its original
        # inference_timestamp, or staleness rejection becomes meaningless.
        original = DetectionBatch(inference_timestamp=100.0, frame_id=7)
        republished = DetectionBatch(
            raw_detections=original.raw_detections,
            selected=original.selected,
            inference_timestamp=original.inference_timestamp,  # not "now"
            frame_id=original.frame_id,
        )
        self.assertEqual(original.inference_timestamp, republished.inference_timestamp)
        self.assertTrue(republished.is_stale(200.0, max_age_s=10.0))


class TestDetectionBatchFromMemory(unittest.TestCase):

    def test_no_detection_this_tick(self):
        batch = detection_batch_from_memory(
            object_class=None, object_confidence=None, object_bbox=None,
            object_distance_mm=None, object_distance_valid=False,
            object_depth_valid_pixel_count=0, object_frame_id=5, object_timestamp=12.0,
        )
        self.assertIsNone(batch.selected)
        self.assertEqual(5, batch.frame_id)
        self.assertEqual(12.0, batch.inference_timestamp)
        self.assertEqual([], batch.raw_detections)

    def test_selected_and_raw_detections(self):
        raw = [
            ("traffic_cone", 0.4, (0, 0, 10, 10), None, False, 0),
            ("traffic_cone", 0.9, (100, 100, 140, 160), 700, True, 55),
        ]
        batch = detection_batch_from_memory(
            object_class="traffic_cone", object_confidence=0.9,
            object_bbox=(100, 100, 140, 160),
            object_distance_mm=700, object_distance_valid=True,
            object_depth_valid_pixel_count=55, object_frame_id=9, object_timestamp=50.0,
            object_raw_detections=raw,
        )
        self.assertEqual("traffic_cone", batch.selected.class_name)
        self.assertEqual(0.9, batch.selected.confidence)
        self.assertEqual(2, len(batch.raw_detections))
        self.assertEqual(0.4, batch.raw_detections[0].confidence)


class TestLaneGeometry(unittest.TestCase):

    def test_stale_when_lost_frames_exceeds_threshold(self):
        fresh = LaneGeometry(yellow_x=100, white_x=200, width_px=100,
                              lost_frames=5, max_accepted_stale_frames=40)
        stale = LaneGeometry(yellow_x=100, white_x=200, width_px=100,
                              lost_frames=41, max_accepted_stale_frames=40)
        self.assertFalse(fresh.stale)
        self.assertTrue(stale.stale)

    def test_visibility_flags(self):
        both = LaneGeometry(yellow_x=100, white_x=200, width_px=100)
        yellow_only = LaneGeometry(yellow_x=100, white_x=None, width_px=100)
        white_only = LaneGeometry(yellow_x=None, white_x=200, width_px=100)
        neither = LaneGeometry(yellow_x=None, white_x=None, width_px=100)

        self.assertTrue(both.both_visible)
        self.assertFalse(both.none_visible)
        self.assertFalse(yellow_only.both_visible)
        self.assertFalse(yellow_only.none_visible)
        self.assertFalse(white_only.both_visible)
        self.assertTrue(neither.none_visible)

    def test_corridor_both_boundaries_visible(self):
        lane = LaneGeometry(yellow_x=100, white_x=200, width_px=100)
        corridor = lane.corridor(vehicle_width_px=50, margin_px=10)
        # center = 150, half_span = max(50, 100)/2 + 10 = 60
        self.assertEqual((90, 210), corridor)

    def test_corridor_yellow_only_uses_white_right_of_yellow_convention(self):
        lane = LaneGeometry(yellow_x=100, white_x=None, width_px=100)
        corridor = lane.corridor(vehicle_width_px=50, margin_px=10, white_right_of_yellow=True)
        # estimated center = 100 + 100/2 = 150, same half_span as above = 60
        self.assertEqual((90, 210), corridor)

    def test_corridor_white_only_is_mirror_of_yellow_only(self):
        lane = LaneGeometry(yellow_x=None, white_x=200, width_px=100)
        corridor = lane.corridor(vehicle_width_px=50, margin_px=10, white_right_of_yellow=True)
        # estimated center = 200 - 100/2 = 150, same as both cases above
        self.assertEqual((90, 210), corridor)

    def test_corridor_none_when_neither_boundary_visible(self):
        lane = LaneGeometry(yellow_x=None, white_x=None, width_px=100)
        self.assertIsNone(lane.corridor(vehicle_width_px=50, margin_px=10))

    def test_corridor_none_when_stale_even_if_boundaries_reported(self):
        # This is the "boundary reported but stale" case round 3 called out:
        # sticky last-known values must not be trusted once lost_frames is high.
        lane = LaneGeometry(yellow_x=100, white_x=200, width_px=100,
                             lost_frames=999, max_accepted_stale_frames=40)
        self.assertIsNone(lane.corridor(vehicle_width_px=50, margin_px=10))

    def test_corridor_none_when_width_missing(self):
        lane = LaneGeometry(yellow_x=100, white_x=200, width_px=None)
        self.assertIsNone(lane.corridor(vehicle_width_px=50, margin_px=10))


class TestLaneGeometryFromMemory(unittest.TestCase):

    def test_defaults_lost_frames_to_zero_when_none(self):
        lane = lane_geometry_from_memory(100, 200, 100, None)
        self.assertEqual(0, lane.lost_frames)

    def test_passes_through_values(self):
        lane = lane_geometry_from_memory(100, 200, 150, 7, max_accepted_stale_frames=20)
        self.assertEqual(100, lane.yellow_x)
        self.assertEqual(200, lane.white_x)
        self.assertEqual(150, lane.width_px)
        self.assertEqual(7, lane.lost_frames)
        self.assertEqual(20, lane.max_accepted_stale_frames)


if __name__ == '__main__':
    unittest.main()
