# Cone Avoidance System

This document explains how the cone-detection and obstacle-avoidance
feature works, how to configure and test it, and what's known to still
need work. It sits on top of the existing `LaneFollower` autopilot
without modifying its core driving behavior.

## Goal

The car normally drives in the right lane of a two-lane (two-way) track,
between the dashed yellow center line and the solid white right edge.
When an orange traffic cone blocks that lane, the car should temporarily
cross into the opposite lane to get around it, then return to its own
lane and resume normal lane-following.

## Architecture

```
cam/image_array ──► ConeDetector ──► object/* (bbox, distance, confidence)
                                           │
lane/yellow_x, lane/white_x ──────────────►│
lane/width_px, lane/lost_frames ──────────►│
                                           ▼
                                   ObstaclePlanner (FSM)
                                           │
                          avoidance/steering, avoidance/throttle,
                          avoidance/command_valid, avoidance/emergency_stop
                                           │
pilot/steering_raw, pilot/throttle_raw ───►│
        (from LaneFollower)                ▼
                                    PilotArbiter
                                           │
                          pilot/steering, pilot/throttle (final)
                                           │
                                           ▼
                                     ObstacleOverlay
                              (draws bbox/corridor/state onto cv/image_array
                               for the web UI - never touches steering/throttle)
```

Each piece is a separate DonkeyCar `Vehicle` part, wired in
`donkeycar/templates/cv_control.py` (or its on-car copy, `manage.py`/
`marcusmanage.py`).

## Files

| File | Purpose |
|---|---|
| `donkeycar/parts/lane_follower.py` | The underlying lane follower (now based on the team's improved `lane_follower3.py` logic). Publishes `lane/yellow_x`, `lane/white_x`, `lane/width_px`, `lane/lost_frames`. |
| `donkeycar/parts/object_detector/opencv_cone_detector.py` | Classical HSV/contour cone detector - no trained model. Finds the cone's bounding box in the camera frame. |
| `donkeycar/parts/object_detector/cone_detector.py` | Wraps the detector backend (`opencv`/`mock`/future `model`), adds cadence/caching and depth-based distance estimation. |
| `donkeycar/parts/obstacle_types.py` | Shared data structures: `Detection`, `LaneGeometry`, `FSMState`, `PassingSide`, corridor math. |
| `donkeycar/parts/obstacle_planner.py` | The decision-making FSM - see below. |
| `donkeycar/parts/pilot_arbiter.py` | The only part allowed to write final `pilot/steering`/`pilot/throttle` - decides whether avoidance output is actually applied, based on `OBSTACLE_AVOIDANCE_MODE`. |
| `donkeycar/parts/obstacle_overlay.py` | Draws debug info (bbox, corridor, FSM state) onto the web UI feed. Diagnostic only. |
| `scripts/hsv_picker.py` | Interactive tool to calibrate `ORANGE_HSV_THRESHOLD_LOW/HIGH` against a real photo of the cone. |

## The FSM (`ObstaclePlanner`)

```
FOLLOW_LANE ──(cone detected in range)──► OBJECT_DETECTED
     ▲                                          │
     │                              (close enough, side has room)
     │                                          ▼
     │                                   PLAN_AVOIDANCE
     │                                          │
     │                            (side picked: LEFT or RIGHT)
     │                                          ▼
     │                                MOVE_AROUND_OBJECT  ─┐
     │                                          │           │  scripted,
     │                                          ▼           │  timed
     │                                    PASS_OBJECT       │  (see below)
     │                                          │           │
     │                                          ▼           │
     │                                    STEER_BACK       ─┘
     │                                          │
     │                                          ▼
     └──────────────────────────────────RETURN_TO_LANE

Any state ──(critical condition)──► EMERGENCY_STOP (stops the car, top priority)
```

- **FOLLOW_LANE**: normal driving, cone not relevant.
- **OBJECT_DETECTED**: a cone is in range. Does **not** force a reduced
  throttle - `LaneFollower`'s own steering/throttle pass straight through
  unmodified (an earlier version did slow down here; user feedback was
  that the baseline cruising speed is already slow enough that a forced
  slowdown right before the swerve did more harm than good). Normal speed
  continues until `MOVE_AROUND_OBJECT` actually commits.
- **PLAN_AVOIDANCE**: cone close enough to commit. Picks LEFT or RIGHT
  based on real-time clearance math (is there enough room on either side
  of the cone within the driving corridor, computed from `lane/yellow_x`/
  `lane/white_x`/`lane/width_px`) - unless `FORCED_PASSING_SIDE` is set
  (see config below), in which case that side is always used and the
  clearance check is skipped entirely for it.
- **MOVE_AROUND_OBJECT / PASS_OBJECT / STEER_BACK**: the actual pass.
  **Deliberately scripted/timed, not continuously re-computed from live
  camera data.** Each phase holds a fixed steering value for a fixed
  duration:
  - `MOVE_AROUND_OBJECT`: steer toward the chosen side (crosses the
    yellow line) for `AVOID_STEER_DURATION_S`.
  - `PASS_OBJECT`: hold straight for `AVOID_STRAIGHT_DURATION_S`. This is
    the blind-spot phase - once alongside/past the cone, the camera
    genuinely cannot see it anymore, so this cannot be confirmation-based.
  - `STEER_BACK`: steer the opposite direction for `AVOID_RETURN_DURATION_S`,
    crossing back to the original lane.

  Throttle during all three phases is **not** fixed either - it's
  `LaneFollower`'s own live throttle (`pilot/throttle_raw`), floored at
  `MIN_MANEUVER_THROTTLE`. The floor exists because `LaneFollower`'s
  throttle can decay toward 0 from its own sustained-lane-loss handling
  right as the swerve engages, which would otherwise turn the wheels
  without the car actually moving; the floor is a minimum, never a boost
  above what `LaneFollower` was already doing.

  This design was a deliberate pivot away from an earlier version that
  continuously recomputed a live steering target from lane geometry
  during the pass - that version repeatedly got confused when a swerve
  brought a *different*, nearby track's lines into view (multiple lanes
  are laid out close together on this course), causing wrong-direction
  steering or getting stuck. The scripted version is immune to that,
  at the cost of being open-loop (not adaptive to exact cone
  size/position).
- **RETURN_TO_LANE**: blends steering from the scripted value toward
  `LaneFollower`'s live output over a few ticks, then hands off control
  completely.
- **EMERGENCY_STOP**: top-priority override. Triggers on: a person
  detected in/near the corridor, the object being critically close, total
  loss of lane visibility while engaged, stale/missing detector data
  during a maneuver, no side having provable clearance (only checked
  during `PLAN_AVOIDANCE`), or the maneuver exceeding its timeout. The
  "critically close" and "stale data" checks are deliberately **not**
  evaluated during `PASS_OBJECT`/`STEER_BACK` (see `BLIND_SPOT_STATES` in
  the code) - a stale cached reading of the cone from right before it
  left the blind spot must not cause a false stop.

  Each of these six checks is individually toggleable in `myconfig.py`
  (all default `True`, i.e. unchanged behavior):
  `ENABLE_PERSON_EMERGENCY_STOP`, `ENABLE_CRITICALLY_CLOSE_EMERGENCY_STOP`,
  `ENABLE_LANE_LOSS_EMERGENCY_STOP`, `ENABLE_NO_DATA_EMERGENCY_STOP`,
  `ENABLE_NO_SAFE_SIDE_EMERGENCY_STOP`, `ENABLE_MANEUVER_TIMEOUT_EMERGENCY_STOP`.
  Leave `PERSON` and `LANE_LOSS` on - a real person near the path, or
  total loss of the track, are the two conditions most likely to matter
  if something genuinely goes wrong. The other four are the ones observed
  to false-positive during testing; disable one only if it's causing more
  trouble than it prevents for a given demo/course.

## Object classes (`CLASS_POLICY` in `myconfig.py`)

The planner's behavior per detected class is a small lookup
(`ObstaclePlanner.DEFAULT_CLASS_POLICY` in `obstacle_planner.py`), not
cone-specific logic - it already distinguishes:

| Class | Behavior |
|---|---|
| `traffic_cone` | Attempts avoidance (the FSM above) |
| `rc_car` | Attempts avoidance, same as a cone |
| `person` | Never avoided - any relevant detection goes straight to `EMERGENCY_STOP`, using a wider `PEDESTRIAN_SAFETY_MARGIN_PX` corridor instead of the normal one |
| `__unknown__` (anything else) | Acknowledged (reaches `OBJECT_DETECTED`) but never attempts avoidance - conservative default |

Override or add classes via `CLASS_POLICY = {"rc_car": {"attempt_avoid": False}, ...}`
in `myconfig.py`.

**Currently inert on real hardware**: `OpenCVConeDetector` (the only
implemented detector backend besides `MockDetector`) always reports
`class_name='traffic_cone'` - it has no way to actually produce a
`person` or `rc_car` detection. The `person`/`rc_car` policy rows only
do anything today when driven through `MockDetector`'s scripted
detections, or once a real trained-model backend
(`CONE_DETECTOR_MODE = "model"`) exists.

## Operating modes (`OBSTACLE_AVOIDANCE_MODE` in `myconfig.py`)

| Mode | What runs | Effect on driving |
|---|---|---|
| `disabled` | Nothing extra | None - pure `LaneFollower` |
| `observe` | `ConeDetector` + overlay only | None - diagnostic only, confirm detection works |
| `shadow` | + `ObstaclePlanner` (full FSM) | None - `PilotArbiter` still passes `LaneFollower`'s output straight through; watch the overlay to see what the planner *would* do |
| `active` | + `PilotArbiter` applies avoidance output | Avoidance can actually steer/throttle |

**Always test in this order**: `observe` → `shadow` → (only once you
trust what you see) → `active`. Each stage is safe by construction - the
mode gate lives entirely in `PilotArbiter`, not scattered through the
FSM.

Any mode other than `disabled` requires `CV_CONTROLLER_CLASS ==
"LaneFollower"` in `myconfig.py` - `cv_control.py`'s `drive()` fails fast
at startup otherwise, since `ObstaclePlanner` depends on the
`lane/yellow_x`, `lane/white_x`, `lane/width_px`, `lane/lost_frames`
outputs only `LaneFollower` publishes.

## Key config values (`myconfig.py` / `cfg_cv_control.py`)

Minimal set to get the feature running at all - see the subsections below
for everything else `cfg_cv_control.py` exposes (it's the source of
truth; every value there has a comment explaining it):

```python
CV_CONTROLLER_MODULE = "donkeycar.parts.lane_follower"
CV_CONTROLLER_CLASS = "LaneFollower"

OBSTACLE_AVOIDANCE_MODE = "observe"   # disabled | observe | shadow | active
CONE_DETECTOR_MODE = "opencv"          # opencv | mock | (model - not implemented)
ORANGE_HSV_THRESHOLD_LOW = (2, 78, 91)     # calibrate with scripts/hsv_picker.py
ORANGE_HSV_THRESHOLD_HIGH = (12, 229, 245)

# Scripted pass maneuver - untuned starting points, tune against real runs:
AVOID_STEER_DURATION_S = 1.0      # phase 1: steering toward the chosen side
AVOID_STRAIGHT_DURATION_S = 1.5   # phase 2: holding straight (blind-spot phase)
AVOID_RETURN_DURATION_S = 1.0     # phase 3: steering back to the original lane
AVOID_STEER_MAGNITUDE = 0.5       # fixed steering magnitude held during phases 1 and 3
AVOID_STEER_POLARITY = 1.0        # flip to -1.0 if avoidance steers the wrong physical direction
MIN_MANEUVER_THROTTLE = 0.15      # floor under LaneFollower's live throttle during the swerve

# None (default) = pick whichever side has more real clearance. "left"/"right"
# forces every pass to one side - see FORCED_PASSING_SIDE below.
FORCED_PASSING_SIDE = None

# Needed only because ~/mycar/config.py is a stale, disconnected copy of
# the repo's cfg_cv_control.py and never got this list added:
from donkeycar.parts.obstacle_types import FSMState
CV_CONTROLLER_OUTPUTS_WITH_ARBITER = ['pilot/steering_raw', 'pilot/throttle_raw', 'cv/image_array',
                                       'lane/yellow_x', 'lane/white_x', 'lane/width_px', 'lane/lost_frames']
```

Tuning guide for the three scripted-maneuver durations: watch
`MOVE_AROUND_OBJECT` - if the car is still mid-turn when it snaps to
straight, raise `AVOID_STEER_DURATION_S`; if it fully crosses before the
timer ends, it can be shortened. Same logic for
`AVOID_STRAIGHT_DURATION_S` (long enough to actually clear the cone's
length) and `AVOID_RETURN_DURATION_S` (long enough to fully cross back).

### `FORCED_PASSING_SIDE`

`None` (default) picks whichever side has more real pixel clearance,
recomputed live each time. Set to `"left"` or `"right"` to always pass on
that side instead, unconditionally (the clearance check is skipped
entirely for the forced side - it's assumed to always have real room).
Useful on a two-lane road where only one side of a cone is ever a real
lane - the other side is just the outer edge/curb, not a second lane,
regardless of what the raw pixel clearance measurement says.

### Detection distance tiers (`ObstaclePlanner`)

Each FSM transition below `PLAN_AVOIDANCE` is gated by one of these
distance thresholds (mm, from the OAK-D's depth stream). Every threshold
has a paired bbox-height-in-pixels fallback, used automatically whenever
`object/distance_valid` is `False` (missing/unreliable depth must never
block class-based detection - see `cone_detector.py`'s
`estimate_distance_mm`):

```python
DETECT_DISTANCE_MM = 3000          # DETECT_BBOX_HEIGHT_PX = 40   -> FOLLOW_LANE -> OBJECT_DETECTED
SLOWDOWN_DISTANCE_MM = 1800        # SLOWDOWN_BBOX_HEIGHT_PX = 70 -> currently unused by any transition (see note below)
AVOID_START_DISTANCE_MM = 1000     # AVOID_START_BBOX_HEIGHT_PX = 110 -> OBJECT_DETECTED -> PLAN_AVOIDANCE
EMERGENCY_STOP_DISTANCE_MM = 400   # EMERGENCY_STOP_BBOX_HEIGHT_PX = 180 -> EMERGENCY_STOP ("critically close")
CLEARED_DISTANCE_MM = 2000         # CLEARED_BBOX_HEIGHT_PX = 60  -> currently unused by any transition (see note below)
```

These are the numbers that decide when the car reacts at all, and are
just as untuned as the maneuver durations above - they were set as
reasonable-sounding starting guesses, not calibrated against real
distance measurements of this course's actual cone placement.
`SLOWDOWN_DISTANCE_MM`/`CLEARED_DISTANCE_MM` are defined and named for a
"slow down as it approaches" / "confirmed cleared" tier but nothing in
`ObstaclePlanner._step_fsm` currently reads them directly - clearing
`OBJECT_DETECTED` back to `FOLLOW_LANE` instead uses the
`CLEAR_CONFIRM_FRAMES` hysteresis counter below (a relevance check, not
a distance check). Worth re-examining if the two seem to disagree during
testing.

### Detection hysteresis / timing (`ObstaclePlanner`)

```python
DETECTION_CONFIRM_FRAMES = 3      # consecutive relevant-detection ticks before FOLLOW_LANE -> OBJECT_DETECTED
CLEAR_CONFIRM_FRAMES = 5          # consecutive not-relevant ticks before considering an object cleared
MAX_NO_DATA_TICKS_DURING_MANEUVER = 10  # consecutive stale/missing-detector ticks during a maneuver -> emergency stop
MANEUVER_TIMEOUT_S = 15.0         # a maneuver stuck longer than this -> emergency stop
DEFAULT_MIN_STATE_DURATION_S = 0.3      # floor on time spent in any FSM state before another transition
MIN_STATE_DURATION_S = {}         # optional per-state override, e.g. {FSMState.MOVE_AROUND_OBJECT: 0.5}
```

### Depth-based distance estimation (`ConeDetector`/`cone_detector.py:estimate_distance_mm`)

Samples a bounded region within the detected bbox (not the full box,
which can include background beside/above the cone) and uses a robust
low percentile rather than a raw minimum, so one noisy depth pixel can't
trigger a false "too close" reading:

```python
DEPTH_ROI_H_FRAC = (0.4, 0.6)   # horizontal slice of the bbox to sample (fraction of width)
DEPTH_ROI_V_FRAC = (0.5, 1.0)   # vertical slice of the bbox to sample (fraction of height)
DEPTH_ROBUST_PERCENTILE = 25    # low percentile used as the distance estimate
DEPTH_MIN_VALID_PIXELS = 20     # below this many valid readings in the ROI, distance is invalid
OBJECT_DETECTOR_MIN_VALID_DEPTH_MM = 200  # readings nearer than this are lens-adjacent noise, not real
OBJECT_DETECTION_HZ = DRIVE_LOOP_HZ       # detector inference cadence; result is cached/republished between ticks
OBJECT_DETECTION_MAX_LATENCY_MS = 500     # cached detection older than this counts as stale ("no data")
```

Still approximate even with the above: the OAK-D's RGB preview and its
stereo-derived depth map aren't perfectly spatially aligned (different
sensors/baseline/FOV - see `donkeycar/parts/oak_d.py`). This narrows the
sampling region, it doesn't solve alignment.

### `PilotArbiter` - final steering/throttle limits

Independent of the FSM's own maneuver logic - a hard backstop applied to
whatever `PilotArbiter` is about to send to the actuators, in `active`
mode:

```python
ARBITER_MAX_STEERING = 1.0
ARBITER_MAX_THROTTLE = THROTTLE_MAX
ARBITER_MIN_THROTTLE = -1.0
MAX_STEERING_RATE_PER_S = 4.0   # hard rate limit, independent of AVOIDANCE_BLEND_STEP
MAX_THROTTLE_RATE_PER_S = 2.0   # emergency stop bypasses this - see PilotArbiter._rate_limit
```

`MAX_STEERING_RATE_PER_S`/`MAX_THROTTLE_RATE_PER_S` matter even if the
planner's own logic is correct: they're the backstop against a planner
bug commanding an instantaneous jump, not just a smoothing nicety.

### Dead/unused config

`THROTTLE_TABLE` is still defined in `cfg_cv_control.py` (and referenced
in `donkeycar/tests/test_obstacle_planner.py`) as a way to override
per-state throttle, but `ObstaclePlanner.__init__` never reads it -
maneuver throttle comes entirely from `MIN_MANEUVER_THROTTLE` +
`LaneFollower`'s live throttle instead (see the FSM section above).
Setting `THROTTLE_TABLE` currently has no effect; either it gets wired up
or it should be removed.

## Calibrating the cone detector

1. Get a real camera frame with the cone in it (pull a frame from a tub,
   or a photo taken with the same camera).
2. `python scripts/hsv_picker.py -f path/to/frame.jpg`
3. Click-drag a box over just the cone body in the window that opens.
4. Press `p` to print the resulting `Mask HSV Low`/`Mask HSV High` values.
5. Put those into `ORANGE_HSV_THRESHOLD_LOW`/`ORANGE_HSV_THRESHOLD_HIGH`.

## Debugging a run

Every FSM state transition and every `EMERGENCY_STOP` trigger is logged
via Python's `logging` module (already configured by `manage.py`). To
capture a clean record of a test run:

```
python manage.py drive --myconfig=myconfig.py 2>&1 | tee run_log.txt
```

Afterward, pull out just the relevant lines (the rest is a noisy
per-200-tick timing table):

```
grep -E "ObstaclePlanner:|No lane line detected" run_log.txt
```

An `EMERGENCY_STOP` log line includes the exact reason, which state it
fired from, how long the car had been in that state, and the detector/
lane data at that moment - enough to diagnose without needing another
test run.

Independent of state-transition logging, `ObstaclePlanner` also emits a
periodic status heartbeat (`STATUS_LOG_INTERVAL_S`, default 1.0s) with
the full picture - state/action/side/blocked, detector reading, lane
width/lost_frames, and the current steering/throttle command - so a run
can be reviewed even across ticks where nothing transitioned.

**Snapshot images**: set `SAVE_OBSTACLE_SNAPSHOTS = True` in
`myconfig.py` and `ObstacleOverlay` periodically (`OBSTACLE_SNAPSHOT_INTERVAL_S`,
default 0.5s) writes the exact annotated overlay frame to disk -
`OBSTACLE_SNAPSHOT_DIR` (default `<car project>/data/obstacle_snapshots`)
- whenever the planner is in any state other than `FOLLOW_LANE`. Each
filename is `<timestamp>_<state>.jpg`, so a saved image can be paired
against the matching `ObstaclePlanner` log lines by timestamp. Off by
default (adds disk I/O) - opt in only when reviewing a specific test run.

## Known limitations / open issues

- **No object identity/tracking.** The system can't distinguish "a stale
  reading of the same cone" from "a genuinely new object" - the
  blind-spot fix (above) trusts that nothing new and critical appears
  during `PASS_OBJECT`/`STEER_BACK`. A second, real obstacle appearing
  during that exact window wouldn't be caught until the car exits the
  blind-spot phase.
- **The three maneuver durations, and the distance-tier thresholds
  (`DETECT_/AVOID_START_/EMERGENCY_STOP_DISTANCE_MM` etc.), are all
  untuned defaults** and will need adjusting for your actual course
  geometry/throttle.
- **`person`/`rc_car` class policies are currently inert.**
  `OpenCVConeDetector` only ever reports `traffic_cone` - the
  stop-immediately `person` handling and the `rc_car` avoidance policy
  (see "Object classes" above) only exercise through `MockDetector` today,
  not on a real drive, until a trained-model detector backend exists.
- **Curves: no longer fully out of scope, but not validated together with
  avoidance.** This note originally said `LaneFollower`'s line tracking
  was significantly less reliable at real curves. That's since improved -
  `LaneFollower` now does multi-row curve-anticipation scanning and
  boundary-identity guards specifically added after a tight-turn failure
  (see `lane_follower.py`'s module docstring). What's *not* yet confirmed
  is the obstacle-avoidance FSM itself (corridor math, the scripted pass)
  behaving correctly while the car is also mid-turn - the corridor
  geometry and the scripted maneuver's fixed steering assume a
  reasonably straight approach. Keep obstacle-avoidance testing on
  straight sections until that combination has specifically been tested.
- **The scripted maneuver is open-loop.** It doesn't adapt to exact cone
  size/position/speed - it assumes a roughly consistent scenario each
  time. A closed-loop version (using lane geometry to confirm lane
  changes, rather than fixed timing) was discussed as a future direction
  but intentionally not built under deadline pressure - see this
  project's git history for the reasoning (continuous live-geometry
  tracking during the pass was the root cause of most bugs fixed along
  the way).
- As of the last test session, a report of "still having a similar
  issue" was not fully diagnosed before time ran out - see `run_log.txt`
  debugging instructions above to capture the exact failure next time.

## Testing checklist for a new session

1. Confirm the car project's copy is up to date: `git log --oneline -1`
   in this repo should show the commit you expect to be running.
2. Confirm `manage.py`'s `ObstaclePlanner` wiring includes
   `pilot/steering_raw` and `pilot/throttle_raw` as inputs (needed for
   the `RETURN_TO_LANE` blend and the `MIN_MANEUVER_THROTTLE` floor,
   respectively).
3. Sanity-check `FORCED_PASSING_SIDE` and the six
   `ENABLE_*_EMERGENCY_STOP` flags in `myconfig.py` are at the values you
   expect for this session - both are easy to leave set from a previous
   test.
4. `OBSTACLE_AVOIDANCE_MODE = "observe"` - confirm the cone bbox appears
   reliably in the web UI overlay.
5. `OBSTACLE_AVOIDANCE_MODE = "shadow"` - confirm `STATE`/`ACTION`/`SIDE`/
   `BLOCKED` in the overlay make sense as a cone is placed/moved, with
   zero effect on actual driving.
6. Only after shadow mode looks correct: `OBSTACLE_AVOIDANCE_MODE =
   "active"`, low speed, e-stop in hand.
