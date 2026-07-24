"""
PilotArbiter - the single chokepoint that decides what actually reaches the
actuators (see donkeycar/parts/obstacle_planner.py and obstacle_types.py).
Added in cv_control.py whenever OBSTACLE_AVOIDANCE_MODE != "disabled".

ObstaclePlanner never writes pilot/steering or pilot/throttle directly; it
publishes avoidance/* candidate commands. PilotArbiter is the only part
that reads both LaneFollower's raw output and the planner's avoidance
output and picks one - this keeps the planner pure/testable and makes
"shadow" mode trivial (the arbiter just ignores avoidance/* there).
"""

import time

from donkeycar.parts.obstacle_types import OperatingMode


class PilotArbiter:

    def __init__(self, cfg):
        self.max_steering = getattr(cfg, 'ARBITER_MAX_STEERING', 1.0)
        self.max_throttle = getattr(cfg, 'ARBITER_MAX_THROTTLE', 1.0)
        self.min_throttle = getattr(cfg, 'ARBITER_MIN_THROTTLE', -1.0)
        # Rate limits are a hard backstop independent of the planner's own
        # blend easing - easing alone doesn't guarantee a bounded rate if
        # the planner has a bug (see design doc point 17).
        self.max_steering_rate_per_s = getattr(cfg, 'MAX_STEERING_RATE_PER_S', 4.0)
        self.max_throttle_rate_per_s = getattr(cfg, 'MAX_THROTTLE_RATE_PER_S', 2.0)

        self._last_time = None
        self._last_steering = 0.0
        self._last_throttle = 0.0

        # Read once at construction, not per tick - OBSTACLE_AVOIDANCE_MODE
        # doesn't change during a drive() session, so there's no need for a
        # dummy Memory key just to shuttle a static string into run().
        self.mode = OperatingMode(getattr(cfg, 'OBSTACLE_AVOIDANCE_MODE', 'disabled'))

    def run(self, pilot_steering_raw, pilot_throttle_raw,
            avoidance_steering, avoidance_throttle, avoidance_command_valid,
            avoidance_emergency_stop):
        now = time.monotonic()
        mode = self.mode

        emergency_bypasses_throttle_rate_limit = False
        if mode == OperatingMode.ACTIVE and avoidance_emergency_stop:
            steering, throttle = 0.0, 0.0
            emergency_bypasses_throttle_rate_limit = True
        elif mode == OperatingMode.ACTIVE and avoidance_command_valid:
            steering, throttle = avoidance_steering, avoidance_throttle
        else:
            # disabled/observe/shadow, or active-but-nothing-to-apply: pure
            # LaneFollower passthrough
            steering = pilot_steering_raw if pilot_steering_raw is not None else 0.0
            throttle = pilot_throttle_raw if pilot_throttle_raw is not None else 0.0

        steering = max(-self.max_steering, min(self.max_steering, steering))
        throttle = max(self.min_throttle, min(self.max_throttle, throttle))

        steering, throttle = self._rate_limit(steering, throttle, now, emergency_bypasses_throttle_rate_limit)
        return steering, throttle

    def _rate_limit(self, steering, throttle, now, bypass_throttle_rate_limit):
        dt = None if self._last_time is None else max(1e-3, now - self._last_time)

        if dt is not None:
            max_dsteer = self.max_steering_rate_per_s * dt
            dsteer = max(-max_dsteer, min(max_dsteer, steering - self._last_steering))
            steering = self._last_steering + dsteer

            if not bypass_throttle_rate_limit:
                max_dthrottle = self.max_throttle_rate_per_s * dt
                dthrottle = max(-max_dthrottle, min(max_dthrottle, throttle - self._last_throttle))
                throttle = self._last_throttle + dthrottle

        self._last_time = now
        self._last_steering = steering
        self._last_throttle = throttle
        return steering, throttle

    def shutdown(self):
        pass
