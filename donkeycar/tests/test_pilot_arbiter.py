import unittest
from unittest.mock import patch

from donkeycar.parts.pilot_arbiter import PilotArbiter


class FakeCfg:
    ARBITER_MAX_STEERING = 1.0
    ARBITER_MAX_THROTTLE = 1.0
    ARBITER_MIN_THROTTLE = -1.0
    MAX_STEERING_RATE_PER_S = 4.0
    MAX_THROTTLE_RATE_PER_S = 2.0
    OBSTACLE_AVOIDANCE_MODE = 'disabled'


def make_arbiter(mode='disabled', **overrides):
    cfg = FakeCfg()
    cfg.OBSTACLE_AVOIDANCE_MODE = mode
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return PilotArbiter(cfg)


class TestPilotArbiter(unittest.TestCase):

    def test_disabled_mode_passes_through_raw_pilot_values(self):
        arbiter = make_arbiter(mode='disabled')
        steering, throttle = arbiter.run(
            pilot_steering_raw=0.3, pilot_throttle_raw=0.2,
            avoidance_steering=0.9, avoidance_throttle=0.9,
            avoidance_command_valid=True, avoidance_emergency_stop=False)
        self.assertAlmostEqual(0.3, steering)
        self.assertAlmostEqual(0.2, throttle)

    def test_observe_mode_ignores_avoidance_output(self):
        arbiter = make_arbiter(mode='observe')
        steering, throttle = arbiter.run(
            0.1, 0.1, avoidance_steering=-0.9, avoidance_throttle=-0.9,
            avoidance_command_valid=True, avoidance_emergency_stop=True)
        self.assertAlmostEqual(0.1, steering)
        self.assertAlmostEqual(0.1, throttle)

    def test_shadow_mode_computes_but_does_not_apply(self):
        arbiter = make_arbiter(mode='shadow')
        steering, throttle = arbiter.run(
            0.05, 0.15, avoidance_steering=0.8, avoidance_throttle=0.05,
            avoidance_command_valid=True, avoidance_emergency_stop=False)
        self.assertAlmostEqual(0.05, steering)
        self.assertAlmostEqual(0.15, throttle)

    def test_active_mode_with_valid_command_applies_avoidance_output(self):
        arbiter = make_arbiter(mode='active')
        steering, throttle = arbiter.run(
            0.0, 0.2, avoidance_steering=0.5, avoidance_throttle=0.1,
            avoidance_command_valid=True, avoidance_emergency_stop=False)
        self.assertAlmostEqual(0.5, steering)
        self.assertAlmostEqual(0.1, throttle)

    def test_active_mode_without_valid_command_falls_back_to_lane_follower(self):
        arbiter = make_arbiter(mode='active')
        steering, throttle = arbiter.run(
            0.2, 0.18, avoidance_steering=0.9, avoidance_throttle=0.9,
            avoidance_command_valid=False, avoidance_emergency_stop=False)
        self.assertAlmostEqual(0.2, steering)
        self.assertAlmostEqual(0.18, throttle)

    def test_active_mode_emergency_stop_forces_zero_regardless_of_avoidance_values(self):
        arbiter = make_arbiter(mode='active')
        steering, throttle = arbiter.run(
            0.0, 0.2, avoidance_steering=0.9, avoidance_throttle=0.9,
            avoidance_command_valid=True, avoidance_emergency_stop=True)
        self.assertAlmostEqual(0.0, steering)
        self.assertAlmostEqual(0.0, throttle)

    def test_none_raw_pilot_values_default_to_zero(self):
        arbiter = make_arbiter(mode='disabled')
        steering, throttle = arbiter.run(
            None, None, avoidance_steering=0.0, avoidance_throttle=0.0,
            avoidance_command_valid=False, avoidance_emergency_stop=False)
        self.assertEqual(0.0, steering)
        self.assertEqual(0.0, throttle)

    def test_output_clamped_to_configured_bounds(self):
        arbiter = make_arbiter(mode='disabled', ARBITER_MAX_STEERING=0.5,
                                ARBITER_MAX_THROTTLE=0.3, ARBITER_MIN_THROTTLE=-0.3)
        steering, throttle = arbiter.run(
            2.0, 5.0, avoidance_steering=0, avoidance_throttle=0,
            avoidance_command_valid=False, avoidance_emergency_stop=False)
        self.assertAlmostEqual(0.5, steering)
        self.assertAlmostEqual(0.3, throttle)

    def test_first_call_is_not_rate_limited(self):
        arbiter = make_arbiter(mode='disabled')
        with patch('donkeycar.parts.pilot_arbiter.time.monotonic', return_value=100.0):
            steering, throttle = arbiter.run(
                1.0, 1.0, avoidance_steering=0, avoidance_throttle=0,
                avoidance_command_valid=False, avoidance_emergency_stop=False)
        self.assertAlmostEqual(1.0, steering)
        self.assertAlmostEqual(1.0, throttle)

    def test_steering_rate_is_limited_on_a_large_jump(self):
        arbiter = make_arbiter(mode='disabled')  # MAX_STEERING_RATE_PER_S = 4.0
        with patch('donkeycar.parts.pilot_arbiter.time.monotonic', return_value=0.0):
            arbiter.run(0.0, 0.0, 0, 0, False, False)
        with patch('donkeycar.parts.pilot_arbiter.time.monotonic', return_value=0.1):
            # requesting a full +1.0 jump in 0.1s; max allowed change is 4.0*0.1=0.4
            steering, _ = arbiter.run(1.0, 0.0, 0, 0, False, False)
        self.assertAlmostEqual(0.4, steering, places=5)

    def test_throttle_rate_is_limited_on_a_large_jump(self):
        arbiter = make_arbiter(mode='disabled')  # MAX_THROTTLE_RATE_PER_S = 2.0
        with patch('donkeycar.parts.pilot_arbiter.time.monotonic', return_value=0.0):
            arbiter.run(0.0, 0.0, 0, 0, False, False)
        with patch('donkeycar.parts.pilot_arbiter.time.monotonic', return_value=0.1):
            # requesting +1.0 jump in 0.1s; max allowed change is 2.0*0.1=0.2
            _, throttle = arbiter.run(0.0, 1.0, 0, 0, False, False)
        self.assertAlmostEqual(0.2, throttle, places=5)

    def test_emergency_stop_bypasses_throttle_rate_limit(self):
        arbiter = make_arbiter(mode='active')
        with patch('donkeycar.parts.pilot_arbiter.time.monotonic', return_value=0.0):
            # get throttle up high first (avoidance flags False -> pure
            # passthrough regardless of mode being 'active')
            arbiter.run(0.0, 1.0, 0, 0, False, False)
        with patch('donkeycar.parts.pilot_arbiter.time.monotonic', return_value=0.01):
            # tiny dt - a rate-limited drop to 0 would barely move, but
            # emergency stop must still land at exactly 0 immediately
            steering, throttle = arbiter.run(
                0.0, 1.0, avoidance_steering=0.0, avoidance_throttle=0.0,
                avoidance_command_valid=True, avoidance_emergency_stop=True)
        self.assertEqual(0.0, throttle)


if __name__ == '__main__':
    unittest.main()
