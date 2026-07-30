"""
ObstacleOverlay - debug visualization for the obstacle-avoidance pipeline
(Milestone 4). Draws the detected object's bbox, the driving corridor, the
avoidance target point, and the planner's FSM state/action/blocked status
on top of cv/image_array - the same image LaneFollower's own OVERLAY_IMAGE
already draws its lane-detection debug info onto (donkeycar/parts/lane_follower.py:overlay_display).

Purely a visualization aid - never touches pilot/steering or pilot/throttle.
Meant to make "shadow" mode actually inspectable in the web UI before
trusting "active" mode: see the design doc's staged rollout (observe ->
shadow -> active).
"""

import logging
import os
import time

import cv2
import numpy as np

from donkeycar.parts.obstacle_types import LaneGeometry

logger = logging.getLogger(__name__)


class ObstacleOverlay:

    def __init__(self, cfg):
        self.vehicle_width_px = getattr(cfg, 'VEHICLE_WIDTH_PX', 60)
        self.corridor_safety_margin_px = getattr(cfg, 'CORRIDOR_SAFETY_MARGIN_PX', 20)
        self.white_right_of_yellow = getattr(cfg, 'WHITE_RIGHT_OF_YELLOW', True)
        self.max_accepted_lane_stale_frames = getattr(cfg, 'MAX_ACCEPTED_LANE_STALE_FRAMES', 40)

        # Where on the frame the corridor/target lines are drawn - roughly
        # the same band LaneFollower's own scan rows occupy, just needs to
        # be somewhere visible; doesn't need to match a real scan row since
        # this is a schematic overlay, not a measurement.
        self.overlay_y = getattr(cfg, 'OBSTACLE_OVERLAY_Y', 130)
        self.overlay_height = getattr(cfg, 'OBSTACLE_OVERLAY_HEIGHT', 30)

        # Optional: save this exact annotated frame to disk periodically
        # while the planner is actively engaged (any state other than
        # FOLLOW_LANE) - lets a test run be reviewed afterward by pairing
        # each saved image's filename/timestamp against the matching
        # ObstaclePlanner log lines (see OBSTACLE_AVOIDANCE.md), instead
        # of relying on a live screenshot at the moment something looks
        # wrong. Off by default (adds disk I/O) - opt in via
        # SAVE_OBSTACLE_SNAPSHOTS = True in myconfig.py.
        self.save_snapshots = getattr(cfg, 'SAVE_OBSTACLE_SNAPSHOTS', False)
        self.snapshot_interval_s = getattr(cfg, 'OBSTACLE_SNAPSHOT_INTERVAL_S', 0.5)
        data_path = getattr(cfg, 'DATA_PATH', '.')
        self.snapshot_dir = getattr(cfg, 'OBSTACLE_SNAPSHOT_DIR',
                                     os.path.join(data_path, 'obstacle_snapshots'))
        self._last_snapshot_at = 0.0
        if self.save_snapshots:
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def run(self, cam_img, object_class, object_confidence, object_bbox,
            object_distance_mm, object_distance_valid,
            lane_yellow_x, lane_white_x, lane_width_px, lane_lost_frames,
            avoidance_state, avoidance_action, avoidance_target_x,
            avoidance_passing_side, avoidance_path_blocked):
        if cam_img is None:
            return None

        img = np.copy(cam_img)
        height = img.shape[0]

        lane = LaneGeometry(yellow_x=lane_yellow_x, white_x=lane_white_x, width_px=lane_width_px,
                             lost_frames=lane_lost_frames or 0,
                             max_accepted_stale_frames=self.max_accepted_lane_stale_frames)
        corridor = lane.corridor(self.vehicle_width_px, self.corridor_safety_margin_px,
                                  self.white_right_of_yellow)

        if corridor is not None:
            c_left, c_right = corridor
            cv2.line(img, (int(c_left), 0), (int(c_left), height), color=(0, 255, 255), thickness=1)
            cv2.line(img, (int(c_right), 0), (int(c_right), height), color=(0, 255, 255), thickness=1)

        if object_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in object_bbox)
            path_blocked = bool(avoidance_path_blocked)
            box_color = (255, 0, 0) if path_blocked else (255, 165, 0)  # red if blocked, orange otherwise
            cv2.rectangle(img, (x1, y1), (x2, y2), color=box_color, thickness=2)
            label = f"{object_class}:{object_confidence:.2f}" if object_confidence is not None else str(object_class)
            if object_distance_valid and object_distance_mm is not None:
                label += f" {object_distance_mm:.0f}mm"
            cv2.putText(img, label, org=(x1, max(0, y1 - 5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4, color=box_color, thickness=1)

        if avoidance_target_x is not None:
            tx = int(avoidance_target_x)
            cv2.line(img, (tx, self.overlay_y), (tx, self.overlay_y + self.overlay_height),
                     color=(255, 0, 255), thickness=2)

        display_str = [
            f"STATE:{avoidance_state}",
            f"ACTION:{avoidance_action}",
            f"SIDE:{avoidance_passing_side}",
            f"BLOCKED:{bool(avoidance_path_blocked)}",
        ]
        y = height - 10 - 10 * len(display_str)
        for s in display_str:
            cv2.putText(img, s, org=(10, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4, color=(255, 255, 255), thickness=1)
            y += 10

        if self.save_snapshots and avoidance_state != 'FOLLOW_LANE':
            now = time.time()
            if now - self._last_snapshot_at >= self.snapshot_interval_s:
                self._last_snapshot_at = now
                filename = f"{now:.2f}_{avoidance_state}.jpg"
                path = os.path.join(self.snapshot_dir, filename)
                # img is RGB (same convention as cam_img throughout this
                # codebase) - cv2.imwrite expects BGR.
                ok = cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if not ok:
                    logger.warning(f"ObstacleOverlay: failed to write snapshot to {path}")

        return img

    def shutdown(self):
        pass
