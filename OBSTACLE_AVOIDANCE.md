# Cone Avoidance System (marcus-object-detection branch)

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
- **OBJECT_DETECTED**: a cone is in range; car slows down.
- **PLAN_AVOIDANCE**: cone close enough to commit. Picks LEFT or RIGHT
  based on real-time clearance math (is there enough room on either side
  of the cone within the driving corridor, computed from `lane/yellow_x`/
  `lane/white_x`/`lane/width_px`).
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

## Key config values (`myconfig.py` / `cfg_cv_control.py`)

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

# Needed only because ~/mycar/config.py is a stale, disconnected copy of
# the repo's cfg_cv_control.py and never got this list added:
from donkeycar.parts.obstacle_types import FSMState
CV_CONTROLLER_OUTPUTS_WITH_ARBITER = ['pilot/steering_raw', 'pilot/throttle_raw', 'cv/image_array',
                                       'lane/yellow_x', 'lane/white_x', 'lane/width_px', 'lane/lost_frames']
```

Tuning guide for the three durations: watch `MOVE_AROUND_OBJECT` -if
the car is still mid-turn when it snaps to straight, raise
`AVOID_STEER_DURATION_S`; if it fully crosses before the timer ends, it
can be shortened. Same logic for `AVOID_STRAIGHT_DURATION_S` (long
enough to actually clear the cone's length) and
`AVOID_RETURN_DURATION_S` (long enough to fully cross back).

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
python marcusmanage.py drive --myconfig=marcusmyconfig.py 2>&1 | tee run_log.txt
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

## Known limitations / open issues

- **No object identity/tracking.** The system can't distinguish "a stale
  reading of the same cone" from "a genuinely new object" - the
  blind-spot fix (above) trusts that nothing new and critical appears
  during `PASS_OBJECT`/`STEER_BACK`. A second, real obstacle appearing
  during that exact window wouldn't be caught until the car exits the
  blind-spot phase.
- **The three maneuver durations are untuned defaults** and will need
  adjusting for your actual course geometry/throttle.
- **Curves are out of scope.** `LaneFollower`'s line tracking is
  significantly less reliable at real curves (yellow/white can swap
  detected order); testing should stay on straight sections.
- **The scripted maneuver is open-loop.** It doesn't adapt to exact cone
  size/position/speed - it assumes a roughly consistent scenario each
  time. A closed-loop version (using lane geometry to confirm lane
  changes, rather than fixed timing) was discussed as a future direction
  but intentionally not built under deadline pressure - see git history
  on this branch for the reasoning (continuous live-geometry tracking
  during the pass was the root cause of most bugs fixed along the way).
- As of the last test session, a report of "still having a similar
  issue" was not fully diagnosed before time ran out - see `run_log.txt`
  debugging instructions above to capture the exact failure next time.

## Testing checklist for a new session

1. Confirm branch is up to date: `git log --oneline -1` should show the
   latest commit on `marcus-object-detection`.
2. Confirm `marcusmanage.py`'s `ObstaclePlanner` wiring includes
   `pilot/steering_raw` as an input (needed for the `RETURN_TO_LANE`
   blend).
3. `OBSTACLE_AVOIDANCE_MODE = "observe"` - confirm the cone bbox appears
   reliably in the web UI overlay.
4. `OBSTACLE_AVOIDANCE_MODE = "shadow"` - confirm `STATE`/`ACTION`/`SIDE`/
   `BLOCKED` in the overlay make sense as a cone is placed/moved, with
   zero effect on actual driving.
5. Only after shadow mode looks correct: `OBSTACLE_AVOIDANCE_MODE =
   "active"`, low speed, e-stop in hand.
