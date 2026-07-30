"""
ObstaclePlanner - decision-only obstacle avoidance FSM (Milestone 3 of the
obstacle-avoidance design; see donkeycar/parts/obstacle_types.py for the
typed data structures this operates on, and donkeycar/parts/pilot_arbiter.py
for the part that actually decides whether any of this reaches the
actuators - ObstaclePlanner never writes pilot/steering or pilot/throttle
itself).

Its core FSM logic (step()) was built and unit-tested standalone, against
hand-built PlannerInput objects, before being wired into cv_control.py -
see run() below for the Vehicle-part entry point, added there whenever
OBSTACLE_AVOIDANCE_MODE is "shadow" or "active".
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from donkeycar.parts.obstacle_types import (
    Action,
    FSMState,
    LaneGeometry,
    OperatingMode,
    PassingSide,
    PlannerDecision,
    PlannerInput,
    AvoidanceCommand,
    Detection,
    detection_batch_from_memory,
    lane_geometry_from_memory,
)

logger = logging.getLogger(__name__)

# States in which the planner is actively engaged with an object (as
# opposed to FOLLOW_LANE, plain cruising with nothing to react to). Lane-
# loss and stale-detector-data emergency checks are scoped to these states
# only - LaneFollower already has its own sustained-lane-loss handling for
# plain cruising, so the planner doubling that up while nothing is even
# being avoided would just be a redundant, premature stop.
ENGAGED_STATES = frozenset([
    FSMState.OBJECT_DETECTED, FSMState.PLAN_AVOIDANCE, FSMState.MOVE_AROUND_OBJECT,
    FSMState.PASS_OBJECT, FSMState.STEER_BACK, FSMState.RETURN_TO_LANE,
])

# States that are part of a committed maneuver and must respect its
# overall timeout, regardless of whether they still depend on live
# corridor geometry (see _check_emergency - only PLAN_AVOIDANCE still
# checks corridor validity/side safety directly; the scripted timed
# phases below deliberately don't, see their class docstring note).
MANEUVER_STATES = frozenset([
    FSMState.PLAN_AVOIDANCE, FSMState.MOVE_AROUND_OBJECT, FSMState.PASS_OBJECT,
    FSMState.STEER_BACK, FSMState.RETURN_TO_LANE,
])

# States where the object being avoided is expected to be genuinely out of
# camera view by design (PASS_OBJECT's own docstring: "the object is in a
# blind spot during this phase"; STEER_BACK likewise, by then behind/beside
# the car on the way back). Checks that depend on freshly re-detecting THAT
# object - "is it still close," "is detector data missing" - must not fire
# here: a lack of fresh data, or a stale-but-still-"fresh" cached reading
# from the object's last real sighting right before it left frame
# (naturally closest/largest then, since the swerve is what put it out of
# view), are the expected condition, not a fault. Confirmed as the cause of
# a premature EMERGENCY_STOP partway through PASS_OBJECT on real track
# testing. Deliberately NOT including MOVE_AROUND_OBJECT - the object may
# still be genuinely in view during that initial turn-in, so its existing
# behavior (see test_stale_detection_during_active_maneuver_eventually_
# triggers_emergency) is unchanged. Also deliberately NOT touching the
# person/stop_immediately check or the maneuver-timeout check below - both
# are state-independent safety nets that must keep running everywhere.
BLIND_SPOT_STATES = frozenset([
    FSMState.PASS_OBJECT, FSMState.STEER_BACK,
])


@dataclass
class ClassPolicy:
    """
    What a class means for the planner - deliberately small, so adding a
    class later (per the design doc) is a label + one row here, not a
    redesign. "else stop"/"no safe side" behavior isn't a per-class field:
    whenever avoidance is attempted and no side is safe, that's always the
    same emergency condition ("no safe passing corridor exists")
    regardless of class - see _check_emergency.
    """
    attempt_avoid: bool = True
    stop_immediately: bool = False       # person: any relevant detection skips the avoidance FSM entirely
    use_pedestrian_corridor: bool = False  # person: relevance uses PEDESTRIAN_SAFETY_MARGIN_PX, not the normal corridor margin


DEFAULT_CLASS_POLICY: Dict[str, ClassPolicy] = {
    "traffic_cone": ClassPolicy(attempt_avoid=True),
    "rc_car": ClassPolicy(attempt_avoid=True),
    "person": ClassPolicy(attempt_avoid=False, stop_immediately=True, use_pedestrian_corridor=True),
    "__unknown__": ClassPolicy(attempt_avoid=False),
}


class _AvoidanceController:
    """
    Converts a corridor-relative target pixel into a raw steering value.

    Deliberately a plain proportional response, not a stateful PID: unlike
    LaneFollower (which has an independent live measurement - the
    detected line's pixel position - that a PID compares against a fixed
    setpoint each frame), there's no equivalent independent measurement
    here. target_x is already a fresh computation from corridor/object
    geometry each time it's set, so a proportional response to
    (target_x - image_center_x) is sufficient and avoids building a PID
    around a feedback signal that doesn't actually exist. Entirely
    separate from LaneFollower's own PID instance - see design doc's
    correctness finding on why that PID can't be shared/mutated.
    """

    def __init__(self, gain: float, output_limit: float = 1.0):
        self.gain = gain
        self.output_limit = output_limit

    def steering_for(self, target_x: float, image_center_x: float) -> float:
        error_px = target_x - image_center_x
        steering = self.gain * error_px
        return max(-self.output_limit, min(self.output_limit, steering))


class ObstaclePlanner:
    """
    Vehicle part (see run()) wrapping the pure step() FSM. Always intended
    to run whenever the obstacle-avoidance feature is enabled - NOT gated
    by run_condition='run_pilot' (see design doc: a part that simply isn't
    called doesn't clear its own internal state). Instead it takes
    pilot_mode as an explicit input and resets itself whenever that's
    False, so re-entering autopilot always starts clean.
    """

    def __init__(self, cfg):
        self.vehicle_width_px = getattr(cfg, 'VEHICLE_WIDTH_PX', 60)
        self.corridor_safety_margin_px = getattr(cfg, 'CORRIDOR_SAFETY_MARGIN_PX', 20)
        self.pedestrian_safety_margin_px = getattr(cfg, 'PEDESTRIAN_SAFETY_MARGIN_PX', 80)
        self.pass_clearance_margin_px = getattr(cfg, 'PASS_CLEARANCE_MARGIN_PX', 10)
        self.white_right_of_yellow = getattr(cfg, 'WHITE_RIGHT_OF_YELLOW', True)

        self.max_accepted_lane_stale_frames = getattr(cfg, 'MAX_ACCEPTED_LANE_STALE_FRAMES', 40)
        self.lane_loss_emergency_ticks = getattr(cfg, 'LANE_LOSS_EMERGENCY_TICKS', 40)
        self.max_latency_s = getattr(cfg, 'OBJECT_DETECTION_MAX_LATENCY_MS', 500) / 1000.0

        self.detection_confirm_frames = getattr(cfg, 'DETECTION_CONFIRM_FRAMES', 3)
        self.clear_confirm_frames = getattr(cfg, 'CLEAR_CONFIRM_FRAMES', 5)
        self.max_no_data_ticks_during_maneuver = getattr(cfg, 'MAX_NO_DATA_TICKS_DURING_MANEUVER', 10)
        self.maneuver_timeout_s = getattr(cfg, 'MANEUVER_TIMEOUT_S', 15.0)
        self.min_state_duration_s = getattr(cfg, 'MIN_STATE_DURATION_S', {}) or {}
        self.default_min_state_duration_s = getattr(cfg, 'DEFAULT_MIN_STATE_DURATION_S', 0.3)

        # Tiered distance thresholds (mm), each paired with a bbox-height
        # (px) equivalent used whenever distance_valid is False - see
        # design doc point 9 (separate detection/slowdown/avoid/emergency/
        # cleared thresholds) and point 2 (never let missing depth block
        # class detection - bbox size is a graceful fallback proxy).
        self.detect_distance_mm = getattr(cfg, 'DETECT_DISTANCE_MM', 3000)
        self.detect_bbox_height_px = getattr(cfg, 'DETECT_BBOX_HEIGHT_PX', 40)
        self.slowdown_distance_mm = getattr(cfg, 'SLOWDOWN_DISTANCE_MM', 1800)
        self.slowdown_bbox_height_px = getattr(cfg, 'SLOWDOWN_BBOX_HEIGHT_PX', 70)
        self.avoid_start_distance_mm = getattr(cfg, 'AVOID_START_DISTANCE_MM', 1000)
        self.avoid_start_bbox_height_px = getattr(cfg, 'AVOID_START_BBOX_HEIGHT_PX', 110)
        self.emergency_stop_distance_mm = getattr(cfg, 'EMERGENCY_STOP_DISTANCE_MM', 400)
        self.emergency_stop_bbox_height_px = getattr(cfg, 'EMERGENCY_STOP_BBOX_HEIGHT_PX', 180)
        self.cleared_distance_mm = getattr(cfg, 'CLEARED_DISTANCE_MM', 2000)
        self.cleared_bbox_height_px = getattr(cfg, 'CLEARED_BBOX_HEIGHT_PX', 60)

        # Throttle per state - reverse disabled by default; EMERGENCY_STOP
        # is always exactly 0.0, not read from this table (see design doc
        # point 16). FOLLOW_LANE isn't here either - command_valid is False
        # in that state, so LaneFollower's own throttle passes straight
        # through the arbiter unmodified.
        default_throttle_table = {
            FSMState.OBJECT_DETECTED: 0.15,
            FSMState.PLAN_AVOIDANCE: 0.12,
            FSMState.MOVE_AROUND_OBJECT: 0.12,
            FSMState.PASS_OBJECT: 0.12,
            FSMState.STEER_BACK: 0.12,
            FSMState.RETURN_TO_LANE: 0.18,
        }
        configured = getattr(cfg, 'THROTTLE_TABLE', {}) or {}
        self.throttle_table = {**default_throttle_table, **configured}

        policy_overrides = getattr(cfg, 'CLASS_POLICY', {}) or {}
        self.class_policy: Dict[str, ClassPolicy] = dict(DEFAULT_CLASS_POLICY)
        for class_name, kwargs in policy_overrides.items():
            self.class_policy[class_name] = ClassPolicy(**kwargs)

        avoidance_gain = getattr(cfg, 'AVOIDANCE_STEERING_GAIN', -0.01)
        # The avoidance controller is open-loop (see its class docstring -
        # no independent live measurement to feed back against), so once
        # it computes a steering value it holds that SAME value, unchanged,
        # for as long as the state persists - there's nothing that
        # tapers it down as the car actually turns. Capping the magnitude
        # here (well below full lock) keeps a held command from being an
        # aggressive full-lock turn that can over-rotate the car off the
        # actual track before the object is confirmed clear (confirmed on
        # real track footage: steering pinned at 1.0 across an entire
        # MOVE_AROUND_OBJECT run put the camera on the building/bushes,
        # off the track, triggering a real lane-loss emergency stop).
        avoidance_max_steering = getattr(cfg, 'AVOIDANCE_MAX_STEERING', 0.5)
        self._avoidance_controller = _AvoidanceController(avoidance_gain, avoidance_max_steering)
        self.avoidance_blend_step = getattr(cfg, 'AVOIDANCE_BLEND_STEP', 0.15)
        self.image_center_x = getattr(cfg, 'IMAGE_W', 320) / 2.0

        # Scripted, timed pass maneuver (MOVE_AROUND_OBJECT / PASS_OBJECT /
        # STEER_BACK below) - replaces continuously recomputing a live
        # corridor/target during the pass. The car deliberately crosses
        # into the opposite lane to get around an object that blocks most
        # of its own lane's width (this is a two-way street, but the
        # object leaves no other option) - once alongside/past the object
        # it's in a genuine blind spot (confirmed by direct observation on
        # the real course), so "wait until the object is confirmed clear"
        # cannot work for this middle phase; these three phases are timed
        # instead. All three are untuned starting points - tune against
        # real runs at your actual throttle.
        self.avoid_steer_duration_s = getattr(cfg, 'AVOID_STEER_DURATION_S', 1.0)
        self.avoid_straight_duration_s = getattr(cfg, 'AVOID_STRAIGHT_DURATION_S', 1.5)
        self.avoid_return_duration_s = getattr(cfg, 'AVOID_RETURN_DURATION_S', 1.0)
        self.avoid_steer_magnitude = getattr(cfg, 'AVOID_STEER_MAGNITUDE', 0.5)
        # Flip to -1.0 on the car if PassingSide.RIGHT ends up steering the
        # car physically left (or vice versa) - avoids needing a code
        # change to correct a real-hardware sign mismatch.
        self.avoid_steer_polarity = getattr(cfg, 'AVOID_STEER_POLARITY', 1.0)

        # Read once at construction, not per tick - OBSTACLE_AVOIDANCE_MODE
        # doesn't change during a drive() session, and step()'s own FSM
        # logic doesn't branch on it anyway (see design doc: mode-based
        # gating is PilotArbiter's job, this is diagnostic-only). Avoids
        # needing a dummy Memory key just to shuttle a static string in.
        self.mode = OperatingMode(getattr(cfg, 'OBSTACLE_AVOIDANCE_MODE', 'disabled'))

        # See run()'s periodic status heartbeat - time.monotonic()-based
        # throttle, independent of state transitions.
        self.status_log_interval_s = getattr(cfg, 'STATUS_LOG_INTERVAL_S', 1.0)
        self._last_status_log_at = 0.0

        self.reset()

    def reset(self):
        self.state = FSMState.FOLLOW_LANE
        self.passing_side = PassingSide.NONE
        self.target_x: Optional[float] = None
        self.emergency_latched = False
        self.emergency_reason = ""
        self.state_entered_at: Optional[float] = None
        self.maneuver_started_at: Optional[float] = None
        self.detect_confirm_count = 0
        self.clear_confirm_count = 0
        self.no_data_count = 0
        self.lane_loss_ticks = 0
        self.blend_ratio = 0.0

    # ------------------------------------------------------------------
    # pure core

    def step(self, pin: PlannerInput) -> Tuple[PlannerDecision, AvoidanceCommand]:
        if not pin.pilot_mode:
            self.reset()
            return (PlannerDecision(state=FSMState.FOLLOW_LANE, action=Action.FOLLOW,
                                     reason="not in pilot mode"),
                    AvoidanceCommand(command_valid=False))

        if self.state_entered_at is None:
            self.state_entered_at = pin.now

        det_batch = pin.detections
        # batch_fresh only asks "is this reading recent enough to trust,"
        # independent of whether it found anything - "found nothing, but
        # the reading is fresh" is genuine clear evidence and must NOT be
        # conflated with "the reading itself is missing/stale" (no
        # evidence either way). See _update_hysteresis.
        batch_fresh = not det_batch.is_stale(pin.now, self.max_latency_s)
        det = det_batch.selected if batch_fresh else None

        lane = pin.lane
        corridor = lane.corridor(self.vehicle_width_px, self.corridor_safety_margin_px, self.white_right_of_yellow)
        # Only meaningful before the car has committed to crossing lanes -
        # once MOVE_AROUND_OBJECT/PASS_OBJECT/STEER_BACK begin, the car is
        # deliberately off in the opposite lane by design (see those
        # states' docstring note), so its own lane's corridor stops being
        # a meaningful safety signal for what's happening right now. Scope
        # this to ENGAGED_STATES minus the scripted phases, so a real,
        # total loss of lane visibility is still caught before/after the
        # scripted pass, without misfiring on the expected "looks like a
        # different lane" view during the scripted phases themselves.
        if self.state in (FSMState.OBJECT_DETECTED, FSMState.PLAN_AVOIDANCE, FSMState.RETURN_TO_LANE):
            self.lane_loss_ticks = self.lane_loss_ticks + 1 if corridor is None else 0
        else:
            self.lane_loss_ticks = 0

        self._update_hysteresis(det, lane, batch_fresh, corridor)

        emergency, reason = self._check_emergency(det, lane, corridor, pin.now)
        if emergency:
            if not self.emergency_latched:
                state_age = pin.now - self.state_entered_at if self.state_entered_at is not None else -1.0
                det_age = det_batch.age(pin.now)
                logger.info(
                    f"ObstaclePlanner: EMERGENCY_STOP triggered from {self.state.value} "
                    f"(in-state {state_age:.2f}s) - reason: {reason} | "
                    f"det={'none' if det is None else f'{det.class_name} conf={det.confidence:.2f} dist={det.distance_mm}'} "
                    f"batch_fresh={batch_fresh} det_age={det_age:.2f}s no_data_count={self.no_data_count} | "
                    f"lane_loss_ticks={self.lane_loss_ticks} corridor={corridor}")
            self.emergency_latched = True
            self.emergency_reason = reason
        if self.emergency_latched:
            self._enter_state(FSMState.EMERGENCY_STOP, pin.now)
            return (PlannerDecision(state=FSMState.EMERGENCY_STOP, action=Action.EMERGENCY_STOP,
                                     path_blocked=True, reason=self.emergency_reason),
                    AvoidanceCommand(steering=0.0, throttle=0.0, command_valid=True, emergency_stop=True))

        return self._step_fsm(det, lane, corridor, pin.now, pin.lane_raw_steering)

    # ------------------------------------------------------------------
    # emergency layer - checked every tick, ahead of normal FSM transitions

    def _check_emergency(self, det, lane, corridor, now) -> Tuple[bool, str]:
        if det is not None:
            policy = self._policy_for(det.class_name)
            if policy.stop_immediately:
                margin = self.pedestrian_safety_margin_px
                ped_corridor = lane.corridor(self.vehicle_width_px, margin, self.white_right_of_yellow)
                # can't prove the person is clear of an expanded corridor we
                # can't even certify - conservative default is to treat them
                # as relevant (see design doc point 10)
                if ped_corridor is None or det.bbox.overlaps_x(*ped_corridor):
                    return True, f"person detected in/near driving corridor (class={det.class_name})"

            # Skipped during the blind-spot phases - see BLIND_SPOT_STATES.
            # A fresh, non-blind-spot "det" here is a real, current
            # sighting and this check still applies normally everywhere
            # else (including MOVE_AROUND_OBJECT, where the object may
            # still be genuinely in view).
            if self.state not in BLIND_SPOT_STATES:
                if self._closer_than(det, self.emergency_stop_distance_mm, self.emergency_stop_bbox_height_px):
                    return True, "object critically close"

        if self.state in ENGAGED_STATES:
            if self.lane_loss_ticks > self.lane_loss_emergency_ticks:
                return True, "lane boundaries lost/stale while engaged with an object"
            # Skipped during the blind-spot phases - see BLIND_SPOT_STATES.
            # A sustained lack of fresh detector data there is the
            # expected condition (the object can't be re-detected by
            # design), not evidence of a detector fault.
            if self.state not in BLIND_SPOT_STATES:
                if self.no_data_count > self.max_no_data_ticks_during_maneuver:
                    return True, "detector data invalid/missing during an active maneuver"

        # Corridor/side-safety only checked in PLAN_AVOIDANCE - that's the
        # only remaining state that depends on live corridor geometry for
        # a decision (see MOVE_AROUND_OBJECT/PASS_OBJECT/STEER_BACK's
        # scripted-timing docstring note for why they deliberately don't).
        if self.state == FSMState.PLAN_AVOIDANCE:
            if corridor is None:
                return True, "no safe passing corridor exists"
            if det is not None:
                clearance_left, clearance_right = self._clearance(det, corridor)
                if self._pick_side(clearance_left, clearance_right) is None:
                    return True, "no side has provable safe clearance"

        if self.state in MANEUVER_STATES:
            if self.maneuver_started_at is not None and (now - self.maneuver_started_at) > self.maneuver_timeout_s:
                return True, "avoidance maneuver exceeded its timeout"

        return False, ""

    # ------------------------------------------------------------------
    # hysteresis bookkeeping

    def _update_hysteresis(self, det, lane, batch_fresh, corridor):
        if not batch_fresh:
            # no information either way - do NOT count this as clearance
            # evidence (that's the point of tracking it separately from
            # clear_confirm_count - see design doc point 6)
            self.no_data_count += 1
            return
        self.no_data_count = 0

        relevant = self._is_relevant(det, lane, corridor)
        if relevant:
            self.detect_confirm_count += 1
            self.clear_confirm_count = 0
        else:
            self.clear_confirm_count += 1
            self.detect_confirm_count = 0

    def _is_relevant(self, det, lane, corridor) -> bool:
        if det is None:
            return False
        policy = self._policy_for(det.class_name)
        if policy.use_pedestrian_corridor:
            # person policy always needs its own, differently-margined
            # corridor, not the normal one passed in.
            corridor = lane.corridor(self.vehicle_width_px, self.pedestrian_safety_margin_px,
                                      self.white_right_of_yellow)
        if corridor is None:
            # can't certify a corridor - conservative: don't silently
            # ignore the detection (design doc point 10)
            return True
        return det.bbox.overlaps_x(*corridor)

    # ------------------------------------------------------------------
    # normal (non-emergency) FSM

    def _step_fsm(self, det, lane, corridor, now, lane_raw_steering=0.0) -> Tuple[PlannerDecision, AvoidanceCommand]:
        """
        Exactly one state transition per call, using an elif chain (not a
        cascade of independent ifs): if a transition happens, this tick
        reports the newly-entered state's action but does NOT also run
        that new state's own transition logic - that happens on the next
        tick, after _check_emergency has had a fresh chance to evaluate
        the new state (this matters most for PLAN_AVOIDANCE: its "no safe
        side" condition is checked by _check_emergency BEFORE _step_fsm's
        PLAN_AVOIDANCE block ever runs, which only works if PLAN_AVOIDANCE
        persists for at least one full tick rather than being entered and
        exited within the same call).
        """
        confirmed_present = self.detect_confirm_count >= self.detection_confirm_frames
        confirmed_clear = self.clear_confirm_count >= self.clear_confirm_frames
        min_duration_elapsed = self._min_duration_elapsed(now)

        if self.state == FSMState.FOLLOW_LANE:
            # Note: NOT gated on the class policy's attempt_avoid here -
            # any relevant, close-enough object should at least be
            # acknowledged (and slowed for) in OBJECT_DETECTED; attempt_avoid
            # only gates whether OBJECT_DETECTED is allowed to proceed on to
            # PLAN_AVOIDANCE below. Gating it here too would mean an
            # unknown-class object (attempt_avoid=False) never even
            # triggers a slow-down, which isn't the intended "conservative"
            # behavior for __unknown__.
            if confirmed_present and det is not None \
                    and self._closer_than(det, self.detect_distance_mm, self.detect_bbox_height_px):
                self._enter_state(FSMState.OBJECT_DETECTED, now)
                return (PlannerDecision(state=FSMState.OBJECT_DETECTED, action=Action.SLOW,
                                         reason="object detected"),
                        self._throttle_only_command(FSMState.OBJECT_DETECTED))
            return (PlannerDecision(state=FSMState.FOLLOW_LANE, action=Action.FOLLOW,
                                     reason="no relevant object in corridor"),
                    AvoidanceCommand(command_valid=False))

        elif self.state == FSMState.OBJECT_DETECTED:
            if confirmed_clear and min_duration_elapsed:
                self._enter_state(FSMState.FOLLOW_LANE, now)
                return (PlannerDecision(state=FSMState.FOLLOW_LANE, action=Action.FOLLOW,
                                         reason="object no longer relevant"),
                        AvoidanceCommand(command_valid=False))
            if det is not None and self._policy_for(det.class_name).attempt_avoid \
                    and self._closer_than(det, self.avoid_start_distance_mm, self.avoid_start_bbox_height_px) \
                    and min_duration_elapsed:
                self._enter_state(FSMState.PLAN_AVOIDANCE, now)
                return (PlannerDecision(state=FSMState.PLAN_AVOIDANCE, action=Action.SLOW,
                                         reason="entering avoidance planning"),
                        self._throttle_only_command(FSMState.PLAN_AVOIDANCE))
            return (PlannerDecision(state=FSMState.OBJECT_DETECTED, action=Action.SLOW,
                                     reason="object detected, not yet ready to plan avoidance"),
                    self._throttle_only_command(FSMState.OBJECT_DETECTED))

        elif self.state == FSMState.PLAN_AVOIDANCE:
            if det is None:
                # No current reading to plan a side from - hold here rather
                # than crash; a sustained gap escalates via no_data_count in
                # _check_emergency on a later tick.
                return (PlannerDecision(state=FSMState.PLAN_AVOIDANCE, action=Action.SLOW,
                                         reason="no current detection while planning"),
                        self._throttle_only_command(FSMState.PLAN_AVOIDANCE))
            # corridor==None and "no side safe" are both already routed to
            # EMERGENCY_STOP by _check_emergency before we get here (this
            # only holds because PLAN_AVOIDANCE persists for a full tick
            # before this block runs - see the docstring above), so a side
            # is guaranteed to exist here.
            clearance_left, clearance_right = self._clearance(det, corridor)
            side = self._pick_side(clearance_left, clearance_right)
            self.passing_side = side
            self.maneuver_started_at = now
            self._enter_state(FSMState.MOVE_AROUND_OBJECT, now)
            return (PlannerDecision(state=FSMState.MOVE_AROUND_OBJECT, action=Action.AVOID,
                                     passing_side=side, path_blocked=True,
                                     clearance_left=clearance_left, clearance_right=clearance_right,
                                     reason="avoidance planned, moving around object"),
                    AvoidanceCommand(steering=self._scripted_steering(side),
                                      throttle=self.throttle_table[FSMState.MOVE_AROUND_OBJECT],
                                      command_valid=True))

        # MOVE_AROUND_OBJECT / PASS_OBJECT / STEER_BACK: a scripted, timed
        # maneuver rather than a continuously recomputed live target - see
        # __init__'s AVOID_*_DURATION_S comment for why. Each phase just
        # holds a fixed steering value for a fixed duration, using
        # self.state_entered_at (reset by _enter_state on every
        # transition) as that phase's own clock; none of them depend on
        # detecting the object again, since PASS_OBJECT is specifically
        # the blind-spot phase where that's not possible.
        elif self.state == FSMState.MOVE_AROUND_OBJECT:
            steering = self._scripted_steering(self.passing_side)
            if (now - self.state_entered_at) >= self.avoid_steer_duration_s:
                self._enter_state(FSMState.PASS_OBJECT, now)
            return (PlannerDecision(state=self.state, action=Action.AVOID,
                                     passing_side=self.passing_side, path_blocked=True,
                                     reason="steering toward chosen side"),
                    AvoidanceCommand(steering=steering, throttle=self.throttle_table[FSMState.MOVE_AROUND_OBJECT],
                                      command_valid=True))

        elif self.state == FSMState.PASS_OBJECT:
            # Held straight - the object is in a blind spot during this
            # phase (confirmed on real footage), so this can only be
            # timed, not confirmed.
            if (now - self.state_entered_at) >= self.avoid_straight_duration_s:
                self._enter_state(FSMState.STEER_BACK, now)
            return (PlannerDecision(state=self.state, action=Action.AVOID,
                                     passing_side=self.passing_side, path_blocked=False,
                                     reason="passing object in blind spot, holding straight"),
                    AvoidanceCommand(steering=0.0, throttle=self.throttle_table[FSMState.PASS_OBJECT],
                                      command_valid=True))

        elif self.state == FSMState.STEER_BACK:
            # Opposite direction from MOVE_AROUND_OBJECT - steering back
            # across to the original lane (white returns to the right,
            # yellow to the left, for the normal WHITE_RIGHT_OF_YELLOW
            # convention).
            steering = self._scripted_steering(self.passing_side, reverse=True)
            if (now - self.state_entered_at) >= self.avoid_return_duration_s:
                # blend_ratio starts at full weight on the scripted steering
                # here and decays to 0 below, so the handoff continues
                # smoothly from whatever STEER_BACK was just holding rather
                # than snapping straight to live lane steering.
                self.blend_ratio = 1.0
                self._enter_state(FSMState.RETURN_TO_LANE, now)
            return (PlannerDecision(state=self.state, action=Action.AVOID,
                                     passing_side=self.passing_side, path_blocked=False,
                                     reason="steering back across to original lane"),
                    AvoidanceCommand(steering=steering, throttle=self.throttle_table[FSMState.STEER_BACK],
                                      command_valid=True))

        elif self.state == FSMState.RETURN_TO_LANE:
            self.blend_ratio = max(0.0, self.blend_ratio - self.avoidance_blend_step)
            # A genuine blend toward LaneFollower's live steering, not just
            # a decay of the avoidance offset toward 0 - see STEER_BACK's
            # comment above. Interpolates from STEER_BACK's held scripted
            # steering value down to whatever LaneFollower currently wants,
            # rather than snapping abruptly to it in one tick.
            steering = (self.blend_ratio * self._scripted_steering(self.passing_side, reverse=True)
                        + (1.0 - self.blend_ratio) * lane_raw_steering)
            if self.blend_ratio <= 0.0 and min_duration_elapsed:
                self._enter_state(FSMState.FOLLOW_LANE, now)
                self.passing_side = PassingSide.NONE
                self.maneuver_started_at = None
                return (PlannerDecision(state=FSMState.FOLLOW_LANE, action=Action.FOLLOW,
                                         reason="returned to lane"),
                        AvoidanceCommand(command_valid=False))
            return (PlannerDecision(state=self.state, action=Action.AVOID,
                                     passing_side=self.passing_side, path_blocked=False,
                                     reason="easing back to lane center"),
                    AvoidanceCommand(steering=steering, throttle=self.throttle_table[FSMState.RETURN_TO_LANE],
                                      command_valid=True))

        # Unreachable in practice, but keeps a defined fallback if a new
        # FSMState is ever added without wiring its branch in here.
        return (PlannerDecision(state=self.state, action=Action.FOLLOW, reason="unhandled state"),
                AvoidanceCommand(command_valid=False))

    # ------------------------------------------------------------------
    # helpers

    def _enter_state(self, new_state: FSMState, now: float):
        if new_state != self.state:
            # Logged (not printed) so it shows up in the same terminal
            # already running manage.py (logging.basicConfig(level=INFO)
            # there) - copy/paste the terminal output after a run to see
            # the exact sequence of transitions, with timing and side,
            # without needing screenshots.
            logger.info(
                f"ObstaclePlanner: {self.state.value} -> {new_state.value} "
                f"(side={self.passing_side.value}, t={now:.2f}s)")
            self.state = new_state
            self.state_entered_at = now

    def _min_duration_elapsed(self, now) -> bool:
        if self.state_entered_at is None:
            return True
        required = self.min_state_duration_s.get(self.state, self.default_min_state_duration_s)
        return (now - self.state_entered_at) >= required

    def _policy_for(self, class_name: str) -> ClassPolicy:
        return self.class_policy.get(class_name, self.class_policy["__unknown__"])

    def _closer_than(self, det: Detection, mm_threshold: float, px_threshold: float) -> bool:
        if det.distance_valid and det.distance_mm is not None:
            return det.distance_mm <= mm_threshold
        return det.bbox.height >= px_threshold

    def _farther_than(self, det: Detection, mm_threshold: float, px_threshold: float) -> bool:
        if det.distance_valid and det.distance_mm is not None:
            return det.distance_mm >= mm_threshold
        return det.bbox.height <= px_threshold

    def _clearance(self, det: Detection, corridor: Tuple[float, float]) -> Tuple[float, float]:
        c_left, c_right = corridor
        return det.bbox.x1 - c_left, c_right - det.bbox.x2

    def _pick_side(self, clearance_left: float, clearance_right: float) -> Optional[PassingSide]:
        required = self.vehicle_width_px + self.pass_clearance_margin_px
        left_ok = clearance_left >= required
        right_ok = clearance_right >= required
        if not left_ok and not right_ok:
            return None
        if left_ok and right_ok:
            return PassingSide.LEFT if clearance_left >= clearance_right else PassingSide.RIGHT
        return PassingSide.LEFT if left_ok else PassingSide.RIGHT

    def _scripted_steering(self, side: PassingSide, reverse: bool = False) -> float:
        """
        Fixed steering magnitude/direction for the scripted pass maneuver
        (MOVE_AROUND_OBJECT steers toward `side`; STEER_BACK calls this
        with reverse=True to steer the opposite way, back across to the
        original lane). AVOID_STEER_POLARITY is a config-only sign flip
        for correcting a real-hardware left/right mismatch without a code
        change - see __init__.
        """
        if side == PassingSide.NONE:
            return 0.0
        sign = 1.0 if side == PassingSide.RIGHT else -1.0
        if reverse:
            sign = -sign
        return self.avoid_steer_polarity * self.avoid_steer_magnitude * sign

    def _throttle_only_command(self, state: FSMState) -> AvoidanceCommand:
        return AvoidanceCommand(steering=0.0, throttle=self.throttle_table.get(state, 0.0), command_valid=True)

    # ------------------------------------------------------------------
    # Vehicle part entry point

    def run(self, object_class, object_confidence, object_bbox, object_center_x,
            object_distance_mm, object_distance_valid, object_depth_valid_pixel_count,
            object_frame_id, object_timestamp, object_raw_detections,
            lane_yellow_x, lane_white_x, lane_width_px, lane_lost_frames,
            run_pilot, pilot_steering_raw=0.0):
        now = time.monotonic()
        detections = detection_batch_from_memory(
            object_class, object_confidence, object_bbox, object_distance_mm,
            object_distance_valid, object_depth_valid_pixel_count,
            object_frame_id, object_timestamp, object_raw_detections)
        lane = lane_geometry_from_memory(lane_yellow_x, lane_white_x, lane_width_px,
                                          lane_lost_frames, self.max_accepted_lane_stale_frames)
        pin = PlannerInput(detections=detections, lane=lane, mode=self.mode,
                            pilot_mode=bool(run_pilot), now=now,
                            lane_raw_steering=pilot_steering_raw if pilot_steering_raw is not None else 0.0)

        decision, command = self.step(pin)

        # Periodic status heartbeat, independent of state transitions -
        # _enter_state only logs when the state actually changes, which
        # says nothing about what the detector/lane data looked like on
        # any given tick in between. This fills that gap: a
        # STATUS_LOG_INTERVAL_S-spaced snapshot of the full picture, so a
        # run can be reviewed after the fact (see OBSTACLE_AVOIDANCE.md)
        # instead of needing a screenshot at the exact moment something
        # looked wrong.
        if now - self._last_status_log_at >= self.status_log_interval_s:
            self._last_status_log_at = now
            det = detections.selected
            det_str = ('none' if det is None else
                       f"{det.class_name} conf={det.confidence:.2f} "
                       f"dist={det.distance_mm} valid={det.distance_valid} "
                       f"bbox_h={det.bbox.height:.0f}")
            logger.info(
                f"ObstaclePlanner status: state={decision.state.value} action={decision.action.value} "
                f"side={decision.passing_side.value} blocked={decision.path_blocked} | "
                f"det={det_str} | lane_width={lane_width_px} lost_frames={lane_lost_frames} | "
                f"steering={command.steering:.2f} throttle={command.throttle:.2f}")

        return (command.steering, command.throttle, command.command_valid, command.emergency_stop,
                decision.state.value, decision.action.value,
                self.target_x, decision.passing_side.value, decision.path_blocked,
                decision.clearance_left, decision.clearance_right, decision.reason)

    def shutdown(self):
        pass
