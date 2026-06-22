import time
import math
from collections import defaultdict
from configs.settings import ACTION_LABELS, ALERT_ACTIONS, HEAVY_MODEL_SKIP
from ultralytics import YOLO


class ActionClassifier:
    """
    Heuristic-based action classification and Pose gesture detection.

    Key improvements over naive approach:
      - Running:  requires sustained high speed for 0.8s, not a single fast frame
      - Fighting: requires boxes deeply overlapping (iou > 0.65, not 0.4)
                  AND at least one person must be moving quickly at the same time
                  (rules out hugging / handshakes / standing close together)
      - Distress: unchanged — both wrists above nose
    """

    # --- Thresholds (tuned for 30 fps webcam) ---
    # Webcam frames are closer to the subject, so pixel displacement per second
    # is higher at the same physical speed.  Values are in pixels/second.
    RUNNING_SPEED_PX        = 300    # px/s — lower than street cam (objects look closer)
    RUNNING_SUSTAIN_SECS    = 0.7    # must stay fast for 0.7 s continuously

    FIGHTING_IOU            = 0.55   # slightly lower than 0.65: webcam perspective flattens depth
    FIGHTING_MIN_SPEED      = 60     # at least one person moving at 60 px/s
    FIGHTING_SUSTAIN_FRAMES = 4      # consistent across 4 frames

    # Fallen: person bounding box is wider than it is tall (lying down)
    FALLEN_ASPECT_RATIO     = 1.35   # width/height  > this = probably lying down
    FALLEN_SUSTAIN_FRAMES   = 6      # must persist for N frames before flagging

    def __init__(self):
        self.history = {}                            # tid → [(timestamp, cx, cy)]
        self._running_start: dict[int, float] = {}  # tid → time when fast movement started
        self._fight_frame_count  = 0                 # consecutive frames where fight condition is met
        self._fallen_counts: dict[int, int] = {}     # tid → consecutive fallen-aspect frames
        self._pose_skip_counter  = 0

        try:
            self.pose_model = YOLO("yolov8n-pose.pt")
            print("[ActionClassifier] Pose model loaded for gesture detection.")
        except Exception as e:
            print(f"Pose model failed: {e}")
            self.pose_model = None

        self.current_frame = None
        print("[ActionClassifier] Heuristic Action Tracker ready.")

    def add_frame(self, frame):
        self.current_frame = frame

    def _speed(self, tid: int, now: float) -> float:
        """Return pixels-per-second for this track over the last 1.5 seconds."""
        hist = self.history.get(tid, [])
        if len(hist) < 4:
            return 0.0
        old_time, old_x, old_y = hist[0]
        new_time, new_x, new_y = hist[-1]
        dt = new_time - old_time
        if dt <= 0:
            return 0.0
        return math.hypot(new_x - old_x, new_y - old_y) / dt

    def classify(self, tracked_persons=None) -> dict:
        if not tracked_persons:
            self._fight_frame_count = 0
            return {"action": "normal", "action_id": 0, "confidence": 1.0, "is_alert": False}

        now = time.time()
        active_ids = []

        detected_action = "normal"
        action_id       = 0
        confidence      = 0.5
        is_alert        = False

        # ── Step 1: Update position history & compute per-person speed ───────
        speeds = {}
        for person in tracked_persons:
            tid = person["track_id"]
            active_ids.append(tid)
            x1, y1, x2, y2 = person["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if tid not in self.history:
                self.history[tid] = []
            self.history[tid].append((now, cx, cy))
            # keep 1.5-second window
            self.history[tid] = [h for h in self.history[tid] if now - h[0] < 1.5]

            speeds[tid] = self._speed(tid, now)

        #  Step 2: Running — sustained fast movement
        #   Must stay above RUNNING_SPEED_PX for RUNNING_SUSTAIN_SECS continuously
        for person in tracked_persons:
            tid = person["track_id"]
            spd = speeds.get(tid, 0.0)

            if spd > self.RUNNING_SPEED_PX:
                if tid not in self._running_start:
                    self._running_start[tid] = now            # start timer
                elif now - self._running_start[tid] >= self.RUNNING_SUSTAIN_SECS:
                    # has been running continuously long enough → flag it
                    detected_action = "running"
                    action_id       = 3
                    confidence      = 0.85
                    is_alert        = True
            else:
                self._running_start.pop(tid, None)            # reset if they slow down

        # ── Step 3: Fighting — deep overlap + movement
        #   Bounding boxes must deeply overlap AND someone must be moving
        #   Rules out: standing close, hugging, handshakes
        fight_condition_met = False
        if len(tracked_persons) >= 2:
            for i in range(len(tracked_persons)):
                for j in range(i + 1, len(tracked_persons)):
                    pa = tracked_persons[i]
                    pb = tracked_persons[j]
                    x1a, y1a, x2a, y2a = pa["bbox"]
                    x1b, y1b, x2b, y2b = pb["bbox"]

                    ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
                    ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)

                    if ix1 < ix2 and iy1 < iy2:
                        overlap_area = (ix2 - ix1) * (iy2 - iy1)
                        area_a = (x2a - x1a) * (y2a - y1a)
                        area_b = (x2b - x1b) * (y2b - y1b)
                        iou = overlap_area / float(area_a + area_b - overlap_area + 1e-6)

                        spd_a = speeds.get(pa["track_id"], 0.0)
                        spd_b = speeds.get(pb["track_id"], 0.0)
                        max_speed = max(spd_a, spd_b)

                        # Deep overlap AND at least one person is moving aggressively
                        if iou > self.FIGHTING_IOU and max_speed > self.FIGHTING_MIN_SPEED:
                            fight_condition_met = True

        if fight_condition_met:
            self._fight_frame_count += 1
        else:
            self._fight_frame_count = max(0, self._fight_frame_count - 1)  # cool down gradually

        if self._fight_frame_count >= self.FIGHTING_SUSTAIN_FRAMES:
            detected_action = "fighting"
            action_id       = 2
            confidence      = 0.95
            is_alert        = True

        # ── Step 4: Fallen person — wide aspect-ratio heuristic ──────────────
        #   A lying/fallen person has a bounding box much wider than tall.
        #   Must persist for FALLEN_SUSTAIN_FRAMES to avoid false positives
        #   from partial detections at frame edges.
        for person in tracked_persons:
            tid = person["track_id"]
            x1, y1, x2, y2 = person["bbox"]
            w = x2 - x1
            h = y2 - y1
            if h > 0 and (w / h) > self.FALLEN_ASPECT_RATIO:
                self._fallen_counts[tid] = self._fallen_counts.get(tid, 0) + 1
                if self._fallen_counts[tid] >= self.FALLEN_SUSTAIN_FRAMES:
                    detected_action = "fallen"
                    action_id       = 6
                    confidence      = 0.88
                    is_alert        = True
            else:
                self._fallen_counts[tid] = max(0, self._fallen_counts.get(tid, 0) - 1)

        # ── Step 5: Distress gesture (Pose — frame-skipped) ──────────────────
        self._pose_skip_counter += 1
        run_pose = (self._pose_skip_counter % HEAVY_MODEL_SKIP == 0)
        if run_pose and self.pose_model is not None and self.current_frame is not None:
            results = self.pose_model(self.current_frame, verbose=False, conf=0.45)[0]
            if (hasattr(results, "keypoints") and results.keypoints is not None
                    and getattr(results.keypoints, "data", None) is not None):
                for kpts in results.keypoints.data:
                    if len(kpts) >= 11:
                        nose_y = float(kpts[0][1])
                        lw_y   = float(kpts[9][1])
                        rw_y   = float(kpts[10][1])
                        nose_c = float(kpts[0][2])
                        lw_c   = float(kpts[9][2])
                        rw_c   = float(kpts[10][2])

                        if nose_c > 0.5 and lw_c > 0.5 and rw_c > 0.5:
                            if lw_y < nose_y and rw_y < nose_y:
                                detected_action = "distress gesture"
                                action_id       = 99
                                confidence      = 0.99
                                is_alert        = True

        # ── Cleanup stale tracks
        for t in [t for t in self.history if t not in active_ids]:
            del self.history[t]
            self._running_start.pop(t, None)
            self._fallen_counts.pop(t, None)

        return {
            "action":     detected_action,
            "action_id":  action_id,
            "confidence": confidence,
            "is_alert":   is_alert
        }
