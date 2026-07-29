"""
Typed data structures shared by the object-detection / obstacle-avoidance
parts: ConeDetector (donkeycar/parts/object_detector/cone_detector.py),
ObstaclePlanner (donkeycar/parts/obstacle_planner.py) and PilotArbiter
(donkeycar/parts/pilot_arbiter.py) - see the design doc for the full
architecture these support.

Vehicle/Memory only stores scalar keys per name, so none of this is wired
through V.add() - each part converts its raw positional inputs into one of
these at the top of run() via the *_from_memory() helpers below, and the
actual FSM/corridor/class-policy logic operates on the typed objects
instead of loose positional arguments.

Coordinate convention (see design doc): all BBox/Detection coordinates are
pixels in the cam/image_array reference frame (cfg.IMAGE_W x IMAGE_H),
origin top-left, x right, y down - the same frame LaneFollower's
lane/yellow_x, lane/white_x already use. A detector model's own input
frame is a different space and must be mapped back via unletterbox_bbox()
before a BBox is ever constructed here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class OperatingMode(str, Enum):
    DISABLED = "disabled"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ACTIVE = "active"


class FSMState(str, Enum):
    FOLLOW_LANE = "FOLLOW_LANE"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    PLAN_AVOIDANCE = "PLAN_AVOIDANCE"
    MOVE_AROUND_OBJECT = "MOVE_AROUND_OBJECT"
    PASS_OBJECT = "PASS_OBJECT"
    RETURN_TO_LANE = "RETURN_TO_LANE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class PassingSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class Action(str, Enum):
    FOLLOW = "follow"
    SLOW = "slow"
    AVOID = "avoid"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned box in the reference frame described in the module
    docstring: pixels, origin top-left, x right, y down."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def overlaps_x(self, lo: float, hi: float) -> bool:
        """True if this box's horizontal extent overlaps the [lo, hi] span at all."""
        return self.x1 < hi and self.x2 > lo


def unletterbox_bbox(x1: float, y1: float, x2: float, y2: float,
                      model_w: float, model_h: float,
                      orig_w: float, orig_h: float) -> BBox:
    """
    Map a bbox from a letterboxed model-input frame back to the original
    camera reference frame (orig_w x orig_h) - see the module docstring on
    why this must happen before a BBox is published anywhere downstream.

    Assumes the standard "uniform resize to fit, center pad" letterbox
    convention: the original image is scaled uniformly (preserving aspect
    ratio) to fit inside model_w x model_h, then centered with padding on
    whichever axis has left-over space.

    input: x1,y1,x2,y2 in model-input pixel space; model_w/model_h (the
           model's input size, e.g. a square 224x224); orig_w/orig_h (the
           camera reference frame size, cfg.IMAGE_W/IMAGE_H)
    output: BBox in the orig_w x orig_h reference frame
    """
    scale = min(model_w / orig_w, model_h / orig_h)
    scaled_w = orig_w * scale
    scaled_h = orig_h * scale
    pad_x = (model_w - scaled_w) / 2.0
    pad_y = (model_h - scaled_h) / 2.0

    def unmap(px, py):
        return (px - pad_x) / scale, (py - pad_y) / scale

    ox1, oy1 = unmap(x1, y1)
    ox2, oy2 = unmap(x2, y2)
    return BBox(ox1, oy1, ox2, oy2)


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: BBox
    distance_mm: Optional[float] = None
    distance_valid: bool = False
    depth_valid_pixel_count: int = 0

    @property
    def center_x(self) -> float:
        return self.bbox.center_x


@dataclass
class DetectionBatch:
    """
    One detector tick's result. `selected` is the single highest-confidence
    detection ObstaclePlanner acts on; `raw_detections` is kept alongside it
    so a wrong planner pick can be told apart from the model missing the
    object entirely (see design doc).

    `inference_timestamp` is time.monotonic() at the moment inference
    actually ran - when ConeDetector caches and re-publishes a result on a
    later drive-loop tick (OBJECT_DETECTION_HZ < DRIVE_LOOP_HZ), this value
    must NOT be refreshed, or staleness rejection (is_stale below) becomes
    meaningless.
    """
    raw_detections: List[Detection] = field(default_factory=list)
    selected: Optional[Detection] = None
    inference_timestamp: Optional[float] = None
    frame_id: int = 0

    def age(self, now: float) -> float:
        """Seconds (monotonic) since this batch was actually inferred."""
        if self.inference_timestamp is None:
            return float('inf')
        return now - self.inference_timestamp

    def is_stale(self, now: float, max_age_s: float) -> bool:
        return self.age(now) > max_age_s


@dataclass
class LaneGeometry:
    """
    lost_frames comes from LaneFollower's new lane/lost_frames output (see
    design doc's "approved touch to lane_follower.py") - the only real
    staleness signal, since lane/yellow_x and lane/white_x are sticky
    last-known values that don't reset when a line is lost.
    """
    yellow_x: Optional[float]
    white_x: Optional[float]
    width_px: Optional[float]
    lost_frames: int = 0
    max_accepted_stale_frames: int = 40

    @property
    def stale(self) -> bool:
        return self.lost_frames > self.max_accepted_stale_frames

    @property
    def both_visible(self) -> bool:
        return self.yellow_x is not None and self.white_x is not None

    @property
    def none_visible(self) -> bool:
        return self.yellow_x is None and self.white_x is None

    def corridor(self, vehicle_width_px: float, margin_px: float,
                 white_right_of_yellow: bool = True) -> Optional[Tuple[float, float]]:
        """
        Derive the driving corridor's (left, right) x-bounds in the
        reference frame, or None if there isn't enough information to
        certify one (design doc's lane-visibility fallback table: stale or
        no boundaries visible -> no corridor, caller must fall back to
        slow/stop rather than guess).

        When only one boundary is visible, the lane center is estimated
        using the same offset convention LaneFollower._lane_center already
        uses (yellow + half width on the side white is expected to be, or
        the mirror image off white) - not a new estimate, a faithful port.
        """
        if self.stale or self.none_visible or self.width_px is None:
            return None

        if self.both_visible:
            center = (self.yellow_x + self.white_x) / 2.0
            # self.width_px is a smoothed/persisted estimate that can lag
            # behind reality (e.g. still narrow from an earlier bad
            # reading, or from before both lines were reacquired together).
            # When both boundaries are actually visible this frame, their
            # live gap is direct evidence of the real lane width right now -
            # taking whichever is WIDER avoids an artificially tight
            # corridor from a stale/undersized smoothed estimate falsely
            # blocking a passable route around an object. Never narrower
            # than the smoothed estimate, so this only ever widens the
            # corridor, never wrongly tightens it.
            effective_width = max(self.width_px, abs(self.white_x - self.yellow_x))
        else:
            sign = 1.0 if white_right_of_yellow else -1.0
            if self.yellow_x is not None:
                center = self.yellow_x + sign * self.width_px / 2.0
            else:
                center = self.white_x - sign * self.width_px / 2.0
            effective_width = self.width_px

        half_span = max(vehicle_width_px, effective_width) / 2.0 + margin_px
        return center - half_span, center + half_span


@dataclass
class PlannerInput:
    detections: DetectionBatch
    lane: LaneGeometry
    mode: OperatingMode
    pilot_mode: bool
    now: float  # time.monotonic()
    # LaneFollower's own current raw steering (pilot/steering_raw) - used
    # only during RETURN_TO_LANE, to genuinely blend toward what
    # LaneFollower currently wants rather than just decaying the
    # avoidance offset toward 0 (see ObstaclePlanner._step_fsm's
    # RETURN_TO_LANE branch). Defaults to 0.0 so existing callers/tests
    # that don't care about the return-to-lane handoff are unaffected.
    lane_raw_steering: float = 0.0


@dataclass
class PlannerDecision:
    state: FSMState
    action: Action
    passing_side: PassingSide = PassingSide.NONE
    path_blocked: bool = False
    clearance_left: Optional[float] = None
    clearance_right: Optional[float] = None
    reason: str = ""


@dataclass
class AvoidanceCommand:
    steering: float = 0.0
    throttle: float = 0.0
    command_valid: bool = False
    emergency_stop: bool = False


def detection_from_values(class_name: Optional[str],
                           confidence: Optional[float],
                           bbox_tuple: Optional[Tuple[float, float, float, float]],
                           distance_mm: Optional[float] = None,
                           distance_valid: bool = False,
                           depth_valid_pixel_count: int = 0) -> Optional[Detection]:
    """
    Build a Detection from the plain values a Memory-backed part would
    hand over (e.g. object/class, object/confidence, object/bbox, ...).
    Returns None if there's nothing to build (no detection this tick) -
    callers should never treat a None class_name/bbox as an error.
    """
    if class_name is None or bbox_tuple is None:
        return None
    x1, y1, x2, y2 = bbox_tuple
    return Detection(
        class_name=class_name,
        confidence=confidence or 0.0,
        bbox=BBox(x1, y1, x2, y2),
        distance_mm=distance_mm,
        distance_valid=bool(distance_valid),
        depth_valid_pixel_count=depth_valid_pixel_count or 0,
    )


def detection_batch_from_memory(object_class, object_confidence, object_bbox,
                                 object_distance_mm, object_distance_valid,
                                 object_depth_valid_pixel_count,
                                 object_frame_id, object_timestamp,
                                 object_raw_detections=None) -> DetectionBatch:
    """
    Assemble a DetectionBatch from the Memory keys ConeDetector publishes.
    `object_raw_detections`, if given, is an iterable of
    (class_name, confidence, bbox_tuple, distance_mm, distance_valid,
    depth_valid_pixel_count) tuples - the same shape detection_from_values
    takes positionally.
    """
    selected = detection_from_values(
        object_class, object_confidence, object_bbox,
        object_distance_mm, object_distance_valid, object_depth_valid_pixel_count,
    )
    raw = []
    for raw_det in (object_raw_detections or []):
        d = detection_from_values(*raw_det)
        if d is not None:
            raw.append(d)
    return DetectionBatch(
        raw_detections=raw,
        selected=selected,
        inference_timestamp=object_timestamp,
        frame_id=object_frame_id or 0,
    )


def lane_geometry_from_memory(lane_yellow_x, lane_white_x, lane_width_px,
                               lane_lost_frames,
                               max_accepted_stale_frames: int = 40) -> LaneGeometry:
    """Assemble a LaneGeometry from LaneFollower's published Memory keys."""
    return LaneGeometry(
        yellow_x=lane_yellow_x,
        white_x=lane_white_x,
        width_px=lane_width_px,
        lost_frames=lane_lost_frames or 0,
        max_accepted_stale_frames=max_accepted_stale_frames,
    )
