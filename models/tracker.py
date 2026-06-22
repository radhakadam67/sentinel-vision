# models/tracker.py
# ByteTrack multi-person tracking via the supervision library
# Assigns persistent IDs to each detected person across frames
# Also handles loitering detection by timing how long each ID stays in a zone

import time
import numpy as np
import supervision as sv
from configs.settings import TRACK_MAX_AGE, LOITER_SECONDS


class PersonTracker:
    """
    Wraps ByteTrack for multi-person tracking.

    ByteTrack works directly off YOLOv8 detections — no separate model file.
    It uses a Kalman filter + IoU matching to keep track IDs stable even
    when a person is briefly occluded.

    Loitering logic:
      Each track ID has a first_seen timestamp.
      If the same ID has been continuously detected for > LOITER_SECONDS,
      it is flagged as loitering.
    """

    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=TRACK_MAX_AGE,
            minimum_matching_threshold=0.8,
            frame_rate=25
        )
        self.track_first_seen: dict[int, float] = {}   # track_id -> timestamp
        print("[Tracker] ByteTrack initialised")

    def update(self, detections: list[dict], frame_shape: tuple) -> list[dict]:
        """
        Update tracker with current frame detections.

        Args:
            detections: Output from ObjectDetector.detect()
            frame_shape: (height, width) of the frame

        Returns:
            List of tracked persons, each as:
            {
                "track_id": int,
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "loitering": bool,
                "duration_seconds": float
            }
        """
        if not detections:
            return []

        # Convert to supervision Detection format
        bboxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confs  = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_ids = np.array([d["class_id"] for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=class_ids
        )

        tracked = self.tracker.update_with_detections(sv_detections)

        now = time.time()
        results = []

        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            bbox = tracked.xyxy[i].tolist()
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            # Track first appearance
            if tid not in self.track_first_seen:
                self.track_first_seen[tid] = now

            duration = now - self.track_first_seen[tid]
            loitering = duration >= LOITER_SECONDS

            results.append({
                "track_id": tid,
                "bbox": [int(v) for v in bbox],
                "confidence": conf,
                "loitering": loitering,
                "duration_seconds": round(duration, 1)
            })

        # Clean up stale track IDs (not seen for > 2x max age in seconds)
        active_ids = {r["track_id"] for r in results}
        stale = [tid for tid in self.track_first_seen if tid not in active_ids]
        for tid in stale:
            del self.track_first_seen[tid]

        return results

    def loitering_score(self, tracked_persons: list[dict]) -> float:
        """
        Returns a 0-1 score based on how many tracked persons are loitering.
        Used by the ensemble fusion layer.
        """
        if not tracked_persons:
            return 0.0
        loitering_count = sum(1 for p in tracked_persons if p["loitering"])
        return min(loitering_count / max(len(tracked_persons), 1), 1.0)
