import types
import unittest

import numpy as np

from donkeycar.parts.obstacle_overlay import ObstacleOverlay


def make_cfg(**overrides):
    cfg = types.SimpleNamespace(
        VEHICLE_WIDTH_PX=60,
        CORRIDOR_SAFETY_MARGIN_PX=20,
        WHITE_RIGHT_OF_YELLOW=True,
        MAX_ACCEPTED_LANE_STALE_FRAMES=40,
        OBSTACLE_OVERLAY_Y=130,
        OBSTACLE_OVERLAY_HEIGHT=30,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestObstacleOverlay(unittest.TestCase):

    def _blank_image(self):
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def test_none_image_returns_none(self):
        overlay = ObstacleOverlay(make_cfg())
        result = overlay.run(None, None, None, None, None, False,
                              100, 250, 150, 0,
                              'FOLLOW_LANE', 'follow', None, 'none', False)
        self.assertIsNone(result)

    def test_output_same_shape_with_no_detection_and_no_lane(self):
        overlay = ObstacleOverlay(make_cfg())
        img = self._blank_image()
        result = overlay.run(img, None, None, None, None, False,
                              None, None, None, 0,
                              'FOLLOW_LANE', 'follow', None, 'none', False)
        self.assertEqual(img.shape, result.shape)
        # no corridor/bbox/target drawn - only the always-on status text
        # block differs from the blank input (top rows should still be
        # untouched, since the status text is anchored near the bottom).
        self.assertTrue(np.array_equal(img[:100], result[:100]))

    def test_corridor_lines_drawn_when_lane_available(self):
        overlay = ObstacleOverlay(make_cfg())
        img = self._blank_image()
        result = overlay.run(img, None, None, None, None, False,
                              100, 250, 150, 0,
                              'FOLLOW_LANE', 'follow', None, 'none', False)
        self.assertFalse(np.array_equal(img, result))  # something was drawn

    def test_bbox_and_label_drawn_when_object_present(self):
        overlay = ObstacleOverlay(make_cfg())
        img = self._blank_image()
        result = overlay.run(img, 'traffic_cone', 0.9, (100, 100, 140, 160), 900, True,
                              None, None, None, 0,
                              'OBJECT_DETECTED', 'slow', None, 'none', False)
        self.assertFalse(np.array_equal(img, result))
        # bbox edge pixel should now be non-zero (rectangle drawn there)
        self.assertTrue(np.any(result[100, 100:141] != 0))

    def test_blocked_object_uses_different_color_than_unblocked(self):
        overlay = ObstacleOverlay(make_cfg())
        img = self._blank_image()
        blocked = overlay.run(img, 'traffic_cone', 0.9, (100, 100, 140, 160), 900, True,
                               None, None, None, 0,
                               'PLAN_AVOIDANCE', 'avoid', None, 'right', True)
        unblocked = overlay.run(img, 'traffic_cone', 0.9, (100, 100, 140, 160), 900, True,
                                 None, None, None, 0,
                                 'PASS_OBJECT', 'avoid', None, 'right', False)
        # top-left corner of the bbox rectangle should differ in color
        self.assertFalse(np.array_equal(blocked[100, 100], unblocked[100, 100]))

    def test_target_x_line_drawn_when_present(self):
        overlay = ObstacleOverlay(make_cfg())
        img = self._blank_image()
        result = overlay.run(img, None, None, None, None, False,
                              None, None, None, 0,
                              'MOVE_AROUND_OBJECT', 'avoid', 200.0, 'right', True)
        self.assertTrue(np.any(result[130:160, 200] != 0))

    def test_state_text_does_not_crash_with_full_realistic_inputs(self):
        overlay = ObstacleOverlay(make_cfg())
        img = self._blank_image()
        result = overlay.run(img, 'traffic_cone', 0.87, (100, 100, 140, 160), 900, True,
                              100, 250, 150, 2,
                              'MOVE_AROUND_OBJECT', 'avoid', 220.0, 'right', True)
        self.assertEqual(img.shape, result.shape)


if __name__ == '__main__':
    unittest.main()
