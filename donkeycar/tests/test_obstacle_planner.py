import types
import unittest

from donkeycar.parts.obstacle_types import (
    BBox,
    Detection,
    DetectionBatch,
    FSMState,
    Action,
    LaneGeometry,
    OperatingMode,
    PassingSide,
    PlannerInput,
)
from donkeycar.parts.obstacle_planner import ObstaclePlanner


def make_cfg(**overrides):
    cfg = types.SimpleNamespace(
        VEHICLE_WIDTH_PX=60,
        CORRIDOR_SAFETY_MARGIN_PX=20,
        PEDESTRIAN_SAFETY_MARGIN_PX=80,
        PASS_CLEARANCE_MARGIN_PX=10,
        WHITE_RIGHT_OF_YELLOW=True,
        MAX_ACCEPTED_LANE_STALE_FRAMES=40,
        LANE_LOSS_EMERGENCY_TICKS=5,
        OBJECT_DETECTION_MAX_LATENCY_MS=500,
        DETECTION_CONFIRM_FRAMES=2,
        CLEAR_CONFIRM_FRAMES=2,
        MAX_NO_DATA_TICKS_DURING_MANEUVER=3,
        MANEUVER_TIMEOUT_S=15.0,
        MIN_STATE_DURATION_S={},
        DEFAULT_MIN_STATE_DURATION_S=0.0,  # no artificial delay in tests unless a test overrides it
        DETECT_DISTANCE_MM=3000, DETECT_BBOX_HEIGHT_PX=40,
        SLOWDOWN_DISTANCE_MM=1800, SLOWDOWN_BBOX_HEIGHT_PX=70,
        AVOID_START_DISTANCE_MM=1000, AVOID_START_BBOX_HEIGHT_PX=110,
        EMERGENCY_STOP_DISTANCE_MM=400, EMERGENCY_STOP_BBOX_HEIGHT_PX=180,
        CLEARED_DISTANCE_MM=2000, CLEARED_BBOX_HEIGHT_PX=60,
        THROTTLE_TABLE={},
        CLASS_POLICY={},
        AVOIDANCE_STEERING_GAIN=-0.01,
        AVOIDANCE_BLEND_STEP=0.5,  # fast blend so tests don't need many ticks
        IMAGE_W=320,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def lane(yellow_x=100, white_x=250, width_px=150, lost_frames=0):
    return LaneGeometry(yellow_x=yellow_x, white_x=white_x, width_px=width_px, lost_frames=lost_frames,
                         max_accepted_stale_frames=40)


def detections(class_name=None, confidence=0.9, bbox=None, distance_mm=None, distance_valid=False,
               timestamp=0.0):
    if class_name is None:
        return DetectionBatch(raw_detections=[], selected=None, inference_timestamp=timestamp, frame_id=1)
    det = Detection(class_name=class_name, confidence=confidence, bbox=BBox(*bbox),
                     distance_mm=distance_mm, distance_valid=distance_valid)
    return DetectionBatch(raw_detections=[det], selected=det, inference_timestamp=timestamp, frame_id=1)


def pin(det_batch, lane_geo, now=0.0, pilot_mode=True, mode=OperatingMode.ACTIVE):
    return PlannerInput(detections=det_batch, lane=lane_geo, mode=mode, pilot_mode=pilot_mode, now=now)


class TestNotInPilotMode(unittest.TestCase):

    def test_resets_and_passes_through(self):
        planner = ObstaclePlanner(make_cfg())
        planner.state = FSMState.MOVE_AROUND_OBJECT  # simulate mid-maneuver
        planner.emergency_latched = True

        decision, command = planner.step(pin(detections(), lane(), pilot_mode=False))

        self.assertEqual(FSMState.FOLLOW_LANE, decision.state)
        self.assertFalse(command.command_valid)
        self.assertEqual(FSMState.FOLLOW_LANE, planner.state)
        self.assertFalse(planner.emergency_latched)


class TestConeAvoidance(unittest.TestCase):
    """
    Cone directly in our corridor (lane 100-250, width 150, corridor with
    margin 20 -> roughly 105-245), close enough to require avoidance, with
    plenty of clearance on the right.
    """

    def _drive_into_avoidance(self, planner, cone_bbox, distance_mm, start_now=0.0):
        now = start_now
        # Exactly one state transition per tick (see ObstaclePlanner._step_fsm):
        # tick1 confirms presence (count 1/2), tick2 confirms presence
        # (count 2/2) and enters OBJECT_DETECTED, tick3 is close enough and
        # enters PLAN_AVOIDANCE, tick4 is where _check_emergency gets a
        # fresh look at PLAN_AVOIDANCE before it transitions to
        # MOVE_AROUND_OBJECT.
        for _ in range(4):
            now += 0.05
            decision, command = planner.step(
                pin(detections('traffic_cone', bbox=cone_bbox, distance_mm=distance_mm, distance_valid=True,
                                timestamp=now),
                    lane(), now=now))
        return decision, command, now

    def test_cone_in_corridor_triggers_avoidance_with_more_room_on_the_right(self):
        planner = ObstaclePlanner(make_cfg())
        # cone bbox (170,180,190,220): x1=170,x2=190 - corridor ~ (105,245).
        # clearance_left = 170-105=65 (< 60+10=70, not enough), clearance_right = 245-190=55 (also not enough)
        # -> pick a bbox with a real safe side instead:
        decision, command, _ = self._drive_into_avoidance(planner, (100, 180, 130, 220), distance_mm=900)

        self.assertEqual(FSMState.MOVE_AROUND_OBJECT, planner.state)
        self.assertEqual(PassingSide.RIGHT, planner.passing_side)  # more room to the right of a cone near the left edge
        self.assertTrue(command.command_valid)
        self.assertGreater(command.throttle, 0)

    def test_no_safe_side_triggers_emergency_stop(self):
        planner = ObstaclePlanner(make_cfg())
        # A wide cone spanning almost the entire ~140px-wide corridor - no
        # side has vehicle_width(60)+margin(10)=70px of clearance.
        decision, command, _ = self._drive_into_avoidance(planner, (115, 180, 235, 220), distance_mm=900)

        self.assertEqual(FSMState.EMERGENCY_STOP, planner.state)
        self.assertTrue(command.emergency_stop)
        self.assertEqual(0.0, command.steering)
        self.assertEqual(0.0, command.throttle)

    def test_cone_outside_corridor_is_ignored(self):
        planner = ObstaclePlanner(make_cfg())
        now = 0.0
        # bbox far to the right of the corridor (corridor ~105-245)
        for _ in range(3):
            now += 0.05
            decision, command = planner.step(
                pin(detections('traffic_cone', bbox=(280, 180, 300, 220), distance_mm=900, distance_valid=True,
                                timestamp=now),
                    lane(), now=now))
        self.assertEqual(FSMState.FOLLOW_LANE, planner.state)
        self.assertFalse(command.command_valid)

    def test_returns_to_lane_after_clearing(self):
        planner = ObstaclePlanner(make_cfg())
        _, _, now = self._drive_into_avoidance(planner, (100, 180, 130, 220), distance_mm=900)
        self.assertEqual(FSMState.MOVE_AROUND_OBJECT, planner.state)

        # object clears - a FRESH reading with nothing detected is genuine
        # clear evidence (not to be confused with a stale/missing reading,
        # which is invalid data - see _update_hysteresis). CLEAR_CONFIRM_FRAMES=2
        # ticks of this -> PASS_OBJECT.
        for _ in range(2):
            now += 0.05
            decision, command = planner.step(pin(detections(timestamp=now), lane(), now=now))
        self.assertEqual(FSMState.PASS_OBJECT, planner.state)

        # another CLEAR_CONFIRM_FRAMES ticks clear -> RETURN_TO_LANE
        for _ in range(2):
            now += 0.05
            decision, command = planner.step(pin(detections(timestamp=now), lane(), now=now))
        self.assertEqual(FSMState.RETURN_TO_LANE, planner.state)

        # blend ratio eases back to 0 (AVOIDANCE_BLEND_STEP=0.5 -> 2 more ticks) -> FOLLOW_LANE
        for _ in range(3):
            now += 0.05
            decision, command = planner.step(pin(detections(timestamp=now), lane(), now=now))
        self.assertEqual(FSMState.FOLLOW_LANE, planner.state)
        self.assertFalse(command.command_valid)
        self.assertEqual(PassingSide.NONE, planner.passing_side)


class TestRcCar(unittest.TestCase):

    def test_rc_car_outside_corridor_is_ignored(self):
        planner = ObstaclePlanner(make_cfg())
        now = 0.0
        for _ in range(3):
            now += 0.05
            decision, command = planner.step(
                pin(detections('rc_car', bbox=(280, 180, 320, 220), distance_mm=1500, distance_valid=True,
                                timestamp=now),
                    lane(), now=now))
        self.assertEqual(FSMState.FOLLOW_LANE, planner.state)
        self.assertFalse(command.command_valid)

    def test_rc_car_entering_corridor_with_room_is_avoided(self):
        planner = ObstaclePlanner(make_cfg())
        now = 0.0
        for _ in range(4):
            now += 0.05
            decision, command = planner.step(
                pin(detections('rc_car', bbox=(100, 180, 130, 220), distance_mm=900, distance_valid=True,
                                timestamp=now),
                    lane(), now=now))
        self.assertEqual(FSMState.MOVE_AROUND_OBJECT, planner.state)
        self.assertEqual(PassingSide.RIGHT, planner.passing_side)


class TestPerson(unittest.TestCase):

    def test_person_in_corridor_triggers_immediate_emergency_stop(self):
        planner = ObstaclePlanner(make_cfg())
        decision, command = planner.step(
            pin(detections('person', bbox=(150, 150, 190, 220), distance_mm=2000, distance_valid=True),
                lane(), now=0.05))
        self.assertEqual(FSMState.EMERGENCY_STOP, planner.state)
        self.assertTrue(command.emergency_stop)

    def test_person_in_expanded_pedestrian_corridor_but_outside_normal_corridor_still_stops(self):
        # normal corridor ~ (105,245); pedestrian corridor (margin 80) is
        # much wider - a person just outside the normal corridor should
        # still trigger a stop.
        planner = ObstaclePlanner(make_cfg())
        decision, command = planner.step(
            pin(detections('person', bbox=(250, 150, 270, 220), distance_mm=2000, distance_valid=True),
                lane(), now=0.05))
        self.assertTrue(command.emergency_stop)

    def test_person_clearly_outside_even_expanded_corridor_does_not_force_a_stop(self):
        planner = ObstaclePlanner(make_cfg())
        decision, command = planner.step(
            pin(detections('person', bbox=(500, 150, 520, 220), distance_mm=2000, distance_valid=True),
                lane(), now=0.05))
        self.assertFalse(command.emergency_stop)
        self.assertEqual(FSMState.FOLLOW_LANE, planner.state)

    def test_never_attempts_to_avoid_a_person(self):
        planner = ObstaclePlanner(make_cfg())
        planner.step(pin(detections('person', bbox=(150, 150, 190, 220), distance_mm=2000, distance_valid=True),
                          lane(), now=0.05))
        self.assertNotEqual(FSMState.MOVE_AROUND_OBJECT, planner.state)
        self.assertNotEqual(PassingSide.LEFT, planner.passing_side)
        self.assertNotEqual(PassingSide.RIGHT, planner.passing_side)


class TestUnknownVsInvalidData(unittest.TestCase):

    def test_unknown_class_is_conservative_but_not_an_automatic_emergency(self):
        planner = ObstaclePlanner(make_cfg())
        now = 0.0
        for _ in range(3):
            now += 0.05
            decision, command = planner.step(
                pin(detections('mystery_object', bbox=(100, 180, 130, 220), distance_mm=900, distance_valid=True,
                                timestamp=now),
                    lane(), now=now))
        # unknown class never attempts avoidance, so it should stay in
        # OBJECT_DETECTED (slowing down), not escalate to an emergency and
        # not swerve.
        self.assertEqual(FSMState.OBJECT_DETECTED, planner.state)
        self.assertFalse(command.emergency_stop)

    def test_invalid_data_while_just_cruising_does_not_force_a_stop(self):
        planner = ObstaclePlanner(make_cfg())
        now = 0.0
        for _ in range(10):
            now += 0.05
            decision, command = planner.step(pin(detections(), lane(), now=now))  # no detection at all, ever
        self.assertEqual(FSMState.FOLLOW_LANE, planner.state)
        self.assertFalse(command.emergency_stop)

    def test_stale_detection_during_active_maneuver_eventually_triggers_emergency(self):
        cfg = make_cfg(MAX_NO_DATA_TICKS_DURING_MANEUVER=2)
        planner = ObstaclePlanner(cfg)
        now = 0.0
        for _ in range(4):
            now += 0.05
            planner.step(pin(detections('traffic_cone', bbox=(100, 180, 130, 220), distance_mm=900,
                                         distance_valid=True, timestamp=now), lane(), now=now))
        self.assertEqual(FSMState.MOVE_AROUND_OBJECT, planner.state)

        # detector goes stale (returns a batch whose inference_timestamp
        # never advances) while mid-maneuver
        stale_batch = DetectionBatch(inference_timestamp=now - 10.0, selected=None)
        for _ in range(4):
            now += 0.05
            decision, command = planner.step(pin(stale_batch, lane(), now=now))

        self.assertEqual(FSMState.EMERGENCY_STOP, planner.state)
        self.assertIn("invalid", decision.reason.lower())


class TestLaneLoss(unittest.TestCase):

    def test_lane_loss_while_cruising_does_not_force_emergency(self):
        # FOLLOW_LANE isn't in ENGAGED_STATES, so plain lane loss while
        # cruising is left to LaneFollower's own handling, not doubled up
        # by the planner.
        planner = ObstaclePlanner(make_cfg())
        now = 0.0
        for _ in range(10):
            now += 0.05
            decision, command = planner.step(pin(detections(), lane(lost_frames=999), now=now))
        self.assertEqual(FSMState.FOLLOW_LANE, planner.state)
        self.assertFalse(command.emergency_stop)

    def test_lane_loss_while_engaged_with_an_object_triggers_emergency(self):
        cfg = make_cfg(LANE_LOSS_EMERGENCY_TICKS=2)
        planner = ObstaclePlanner(cfg)
        now = 0.0
        # get into OBJECT_DETECTED first (not yet close enough to plan avoidance)
        for _ in range(3):
            now += 0.05
            decision, command = planner.step(
                pin(detections('traffic_cone', bbox=(150, 180, 170, 200), distance_mm=1500, distance_valid=True,
                                timestamp=now),
                    lane(), now=now))
        self.assertEqual(FSMState.OBJECT_DETECTED, planner.state)

        # now lose the lane while still engaged
        for _ in range(4):
            now += 0.05
            decision, command = planner.step(
                pin(detections('traffic_cone', bbox=(150, 180, 170, 200), distance_mm=1500, distance_valid=True,
                                timestamp=now),
                    lane(yellow_x=None, white_x=None), now=now))

        self.assertEqual(FSMState.EMERGENCY_STOP, planner.state)


class TestManeuverTimeout(unittest.TestCase):

    def test_maneuver_exceeding_timeout_triggers_emergency(self):
        cfg = make_cfg(MANEUVER_TIMEOUT_S=0.2, CLEAR_CONFIRM_FRAMES=1000)  # never naturally clears
        planner = ObstaclePlanner(cfg)
        now = 0.0
        for _ in range(4):
            now += 0.05
            planner.step(pin(detections('traffic_cone', bbox=(100, 180, 130, 220), distance_mm=900,
                                         distance_valid=True, timestamp=now), lane(), now=now))
        self.assertEqual(FSMState.MOVE_AROUND_OBJECT, planner.state)

        now += 1.0  # well past the 0.2s timeout, object still "present"
        decision, command = planner.step(
            pin(detections('traffic_cone', bbox=(100, 180, 130, 220), distance_mm=900, distance_valid=True,
                            timestamp=now), lane(), now=now))
        self.assertEqual(FSMState.EMERGENCY_STOP, planner.state)


class TestBboxFallbackWhenDepthInvalid(unittest.TestCase):

    def test_uses_bbox_height_when_distance_invalid(self):
        planner = ObstaclePlanner(make_cfg())
        now = 0.0
        # no distance_mm/distance_valid at all - bbox height (220-180=40 -> below
        # AVOID_START_BBOX_HEIGHT_PX=110) should NOT be enough to plan avoidance yet
        for _ in range(3):
            now += 0.05
            decision, command = planner.step(
                pin(detections('traffic_cone', bbox=(100, 180, 130, 220), distance_mm=None, distance_valid=False,
                                timestamp=now),
                    lane(), now=now))
        self.assertEqual(FSMState.OBJECT_DETECTED, planner.state)

        # a much taller bbox (height=150 >= 110) should trigger avoidance via the proxy
        for _ in range(3):
            now += 0.05
            decision, command = planner.step(
                pin(detections('traffic_cone', bbox=(100, 60, 130, 210), distance_mm=None, distance_valid=False,
                                timestamp=now),
                    lane(), now=now))
        self.assertEqual(FSMState.MOVE_AROUND_OBJECT, planner.state)


class TestFeatureDisabledViaCallerNotWiring(unittest.TestCase):

    def test_step_is_a_pure_function_independent_of_mode_field(self):
        # ObstaclePlanner's own FSM logic doesn't branch on PlannerInput.mode
        # (see design doc: mode-based gating is PilotArbiter's job) - it's
        # only used for diagnostics/logging. Confirm SHADOW vs ACTIVE
        # produce identical decisions/commands from the planner's own POV.
        cfg = make_cfg()
        planner_shadow = ObstaclePlanner(cfg)
        planner_active = ObstaclePlanner(cfg)
        now = 0.0
        for _ in range(3):
            now += 0.05
            d1, c1 = planner_shadow.step(
                pin(detections('traffic_cone', bbox=(100, 180, 130, 220), distance_mm=900, distance_valid=True,
                                timestamp=now),
                    lane(), now=now, mode=OperatingMode.SHADOW))
            d2, c2 = planner_active.step(
                pin(detections('traffic_cone', bbox=(100, 180, 130, 220), distance_mm=900, distance_valid=True,
                                timestamp=now),
                    lane(), now=now, mode=OperatingMode.ACTIVE))
        self.assertEqual(d1.state, d2.state)
        self.assertEqual(c1.steering, c2.steering)
        self.assertEqual(c1.command_valid, c2.command_valid)


if __name__ == '__main__':
    unittest.main()
