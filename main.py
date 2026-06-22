"""
main.py  —  Surveillance AI real-time pipeline
==================================================
Run:
    python main.py               # uses CAMERA_SOURCE from configs/settings.py
    python main.py --source 0    # override to webcam 0
    python main.py --source rtsp://your_ip/stream
    python main.py --source /path/to/video.mp4

Keys:
    Q  — quit
    S  — force-save a snapshot right now
    R  — reset anomaly baseline (recalibrate)
"""

import cv2
import time
import os
import sys
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import ObjectDetector
from models.tracker import PersonTracker
from models.action_classifier import ActionClassifier
from models.anomaly_detector import AnomalyDetector
from models.weapon_detector import WeaponDetector
from models.ensemble import EnsembleFusion
from utils.alert import AlertDispatcher
from configs.settings import (
    CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, HEAVY_MODEL_SKIP, ANOMALY_WARMUP_FRAMES
)

try:
    from models.face_recognizer import FaceWatchlist
except ImportError:
    FaceWatchlist = None

# ---------------------------------------------------------------------------
# Threaded frame reader  — decouples camera I/O from processing so we never
# block on a slow network stream or USB latency
# ---------------------------------------------------------------------------

class FrameReader:
    """Background thread that reads frames continuously from the camera."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        # Reduce OpenCV internal buffer to 1 → always get the LATEST frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame  = None
        self._ok     = False
        self._lock   = threading.Lock()
        self._stopped = False

        if not self.cap.isOpened():
            raise RuntimeError(f"[FrameReader] Cannot open source: {source}")

        # Prime with one read before starting thread
        self._ok, self._frame = self.cap.read()
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        print(f"[FrameReader] Streaming from: {source}")

    def _reader(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            with self._lock:
                self._ok    = ok
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def release(self):
        self._stopped = True
        time.sleep(0.1)
        self.cap.release()


# ---------------------------------------------------------------------------
# Alert banner state
# ---------------------------------------------------------------------------

alert_banner = {
    "active":     False,
    "message":    "",
    "reason":     "",
    "timestamp":  "",
    "show_until": 0.0,
}
BANNER_DURATION = 8   # seconds


def show_alert_banner(frame, banner: dict) -> None:
    """Draws a bold red-gradient alert banner across the top of the frame."""
    if not banner["active"] or time.time() > banner["show_until"]:
        banner["active"] = False
        return

    h, w = frame.shape[:2]

    # Gradient-like dark red background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 94), (10, 0, 160), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Pulsing red border
    if int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    # ALERT label
    cv2.putText(frame, "ALERT", (14, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3)

    # Event type
    cv2.putText(frame, banner["message"], (130, 42),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 80, 255), 2)

    # Reason detail
    cv2.putText(frame, banner["reason"], (14, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    # Timestamp (top right)
    cv2.putText(frame, banner["timestamp"], (w - 205, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


def trigger_alert_banner(fusion_result: dict):
    """Set banner state when ensemble fires an alert."""
    reasons = fusion_result.get("alert_reasons", [])
    action  = fusion_result.get("action", "")

    reason_text = reasons[0] if reasons else f"Risk: {fusion_result['risk_score']:.0%}"

    if any("WATCHLIST" in r for r in reasons):
        message = "⚠ WATCHLIST MATCH"
        reason_text = next((r for r in reasons if "WATCHLIST" in r), reason_text)
    elif any("WEAPON" in r for r in reasons):
        message = "⚠ WEAPON — CRITICAL"
        reason_text = next((r for r in reasons if "WEAPON" in r), reason_text)
    elif "fighting" in action:
        message = "FIGHTING DETECTED"
    elif "strike" in action:
        message = "⚠ STRIKE / PUNCH"
    elif "kick" in action:
        message = "⚠ KICK DETECTED"
    elif "throwing" in action:
        message = "THROWING DETECTED"
    elif "fallen" in action:
        message = "⚠ PERSON FALLEN"
    elif "distress" in action:
        message = "⚠ DISTRESS — HANDS UP"
    elif "loitering" in action or any("loiter" in r.lower() for r in reasons):
        message = "LOITERING DETECTED"
    elif "robbery" in action:
        message = "ROBBERY DETECTED"
    elif "vandalism" in action:
        message = "VANDALISM DETECTED"
    elif "running" in action:
        message = "RUNNING DETECTED"
    elif fusion_result.get("anomaly_score", 0) > 0.65:
        message = "SCENE ANOMALY"
    else:
        message = "SUSPICIOUS BEHAVIOUR"

    if len(reason_text) > 72:
        reason_text = reason_text[:69] + "..."

    alert_banner["active"]     = True
    alert_banner["message"]    = message
    alert_banner["reason"]     = reason_text
    alert_banner["timestamp"]  = datetime.now().strftime("%H:%M:%S")
    alert_banner["show_until"] = time.time() + BANNER_DURATION


# ---------------------------------------------------------------------------
# HUD / annotations
# ---------------------------------------------------------------------------

def draw_calibration_overlay(frame, frame_count: int):
    """Show a progress bar while the anomaly detector is calibrating."""
    h, w = frame.shape[:2]
    progress = min(frame_count / ANOMALY_WARMUP_FRAMES, 1.0)
    bar_w    = 280
    filled   = int(bar_w * progress)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(frame, "CALIBRATING SCENE BASELINE...", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    cv2.rectangle(frame, (w - bar_w - 14, 14), (w - 14, 36), (60, 60, 60), -1)
    cv2.rectangle(frame, (w - bar_w - 14, 14), (w - bar_w - 14 + filled, 36),
                  (0, 200, 100), -1)
    cv2.putText(frame, f"{progress:.0%}", (w - bar_w // 2 - 18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_annotations(frame, tracked_persons, fusion_result, action_result,
                      fps: float, calibrating: bool, frame_count: int):
    """Draw bounding boxes, track IDs, severity bar, weapon boxes, and action label."""
    h, w = frame.shape[:2]

    # Calibration overlay (drawn first, under everything)
    if calibrating:
        draw_calibration_overlay(frame, frame_count)

    # --- Bounding boxes per person ---
    for person in tracked_persons:
        x1, y1, x2, y2 = person["bbox"]
        tid       = person["track_id"]
        loitering = person["loitering"]
        duration  = person["duration_seconds"]

        color = (0, 0, 220) if loitering else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{tid}"
        if loitering:
            label += f"  LOITERING {duration:.0f}s"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # --- Weapon bounding boxes (bright orange) ---
    for bbox in fusion_result.get("weapon_boxes", []):
        wx1, wy1, wx2, wy2 = bbox
        cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 128, 255), 3)
        cv2.putText(frame, "WEAPON", (wx1, wy1 - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 128, 255), 2)

    # --- Risk score bar + severity label (bottom left) ---
    risk     = fusion_result["risk_score"]
    severity = fusion_result.get("severity", "LOW")
    bar_w    = 240
    bar_fill = int(bar_w * risk)
    severity_colors = {
        "LOW":      (0,  200,   0),
        "MEDIUM":   (0,  200, 255),
        "HIGH":     (0,  140, 255),
        "CRITICAL": (0,    0, 255),
    }
    bar_color = severity_colors.get(severity, (0, 200, 0))
    cv2.rectangle(frame, (10, h - 64), (10 + bar_w, h - 42), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h - 64), (10 + bar_fill, h - 42), bar_color, -1)
    cv2.putText(frame, f"Risk: {risk:.0%}  [{severity}]", (14, h - 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # --- Crowd score ---
    crowd = fusion_result.get("crowd_score", 0.0)
    if crowd > 0:
        cv2.putText(frame, f"Crowd risk: {crowd:.0%}", (14, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # --- Anomaly score (bottom left, 2nd line) ---
    anomaly = fusion_result.get("anomaly_score", 0.0)
    anom_color = (0, 80, 255) if anomaly > 0.6 else (150, 150, 150)
    cv2.putText(frame, f"Anomaly: {anomaly:.0%}", (14, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, anom_color, 1)

    # --- Person count (bottom right) ---
    cv2.putText(frame, f"Persons: {fusion_result['person_count']}", (w - 150, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # --- FPS (top right) ---
    fps_color = (0, 200, 0) if fps >= 20 else (0, 100, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)

    # --- Action label ---
    action = action_result.get("action", "")
    ACTION_COLORS = {
        "fighting":        (0,   0, 255),
        "strike":          (0,  80, 255),
        "kick":            (0, 100, 255),
        "throwing":        (0, 160, 255),
        "fallen":          (0,   0, 200),
        "distress gesture":(0,   0, 220),
        "running":         (0, 200, 255),
    }
    banner_offset = 100 if calibrating else 0
    if action not in ("normal", "buffering", ""):
        color = ACTION_COLORS.get(action, (0, 100, 255))
        cv2.putText(frame, f"Action: {action.upper()}", (10, 112 + banner_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- Alert banner drawn last (on top of everything) ---
    show_alert_banner(frame, alert_banner)

    return frame


def save_snapshot(frame, reason: str, snap_dir: str = "snapshots") -> str:
    """Save alert frame as JPEG evidence."""
    os.makedirs(snap_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(snap_dir, f"alert_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[Snapshot] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Surveillance AI")
    parser.add_argument("--source", default=None,
                        help="Camera source: 0/1 (webcam), rtsp://…, or video file path")
    args = parser.parse_args()

    # Resolve source: CLI > settings.py CAMERA_SOURCE
    raw_source = args.source if args.source is not None else CAMERA_SOURCE
    # Convert "0", "1"… strings to ints for webcam selection
    try:
        source = int(raw_source)
    except (ValueError, TypeError):
        source = raw_source

    print("\n=== Surveillance AI — Starting pipeline ===\n")

    detector   = ObjectDetector()
    tracker    = PersonTracker()
    classifier = ActionClassifier()
    anomaly    = AnomalyDetector()
    weapon     = WeaponDetector()
    ensemble   = EnsembleFusion()
    alerter    = AlertDispatcher()

    face_recognizer = None
    face_executor   = None
    face_task       = None
    current_watchlist_matches: list[str] = []

    if FaceWatchlist is not None:
        try:
            face_recognizer = FaceWatchlist(watchlist_dir="watchlist_faces")
            face_executor   = ThreadPoolExecutor(max_workers=1)
        except Exception as e:
            print(f"[FaceWatchlist] Failed to load: {e}")
    else:
        print("[FaceWatchlist] facenet-pytorch not installed — watchlist disabled")

    # --- Open threaded frame reader ---
    try:
        reader = FrameReader(source)
    except RuntimeError as e:
        print(e)
        return

    print(f"[Pipeline] Running — press Q to quit, S to snapshot, R to reset baseline\n")

    frame_count      = 0
    weapon_skip_ctr  = 0
    last_fps_time    = time.time()
    fps_display      = 0.0
    last_weapon      = {"weapon_detected": False, "weapons": [], "weapon_boxes": [], "weapon_score": 0.0}
    calibrating      = True

    while True:
        ret, frame = reader.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        frame_count += 1

        # FPS calculation (update every 30 frames)
        if frame_count % 30 == 0:
            fps_display  = 30 / max(time.time() - last_fps_time, 1e-6)
            last_fps_time = time.time()

        # --- Detection + tracking (every frame) ---
        detections      = detector.detect(frame)
        tracked_persons = tracker.update(detections, frame.shape[:2])

        # --- Action classification (every frame, uses pose every N frames) ---
        classifier.add_frame(frame)
        action_result   = classifier.classify(tracked_persons)

        # --- Anomaly detection (every frame, but suppressed during warmup) ---
        anomaly_result  = anomaly.score(frame)
        calibrating     = anomaly_result.get("calibrating", False)

        # --- Weapon detection (every HEAVY_MODEL_SKIP frames) ---
        weapon_skip_ctr += 1
        if weapon_skip_ctr % HEAVY_MODEL_SKIP == 0:
            last_weapon = weapon.detect(frame)

        # --- Ensemble fusion ---
        fusion_result = ensemble.fuse(
            tracked_persons, action_result, anomaly_result, last_weapon
        )

        # --- Face watchlist (async, every frame) ---
        if face_recognizer is not None:
            if face_task is None or face_task.done():
                if face_task is not None:
                    try:
                        current_watchlist_matches = face_task.result()
                    except Exception:
                        current_watchlist_matches = []
                face_task = face_executor.submit(
                    face_recognizer.check_persons, frame.copy(), tracked_persons
                )
            if current_watchlist_matches:
                fusion_result["should_alert"] = True
                fusion_result["risk_score"]   = 1.0
                fusion_result["alert_reasons"].insert(
                    0, f"WATCHLIST MATCH: {', '.join(current_watchlist_matches)}"
                )

        # --- Alert dispatch ---
        if fusion_result["should_alert"]:
            trigger_alert_banner(fusion_result)
            snapshot_path = save_snapshot(frame, str(fusion_result["alert_reasons"]))
            alerter.dispatch(fusion_result, snapshot_path)

        # --- Draw annotations ---
        annotated = draw_annotations(
            frame.copy(), tracked_persons, fusion_result, action_result,
            fps_display, calibrating, frame_count
        )

        cv2.imshow("Surveillance AI", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[Pipeline] Stopped by user")
            break
        elif key == ord("s"):
            save_snapshot(frame, "manual")
        elif key == ord("r"):
            # Reset anomaly baseline
            anomaly._warmup_losses.clear()
            anomaly._calibrated         = False
            anomaly._dynamic_threshold  = None
            calibrating                 = True
            print("[Pipeline] Anomaly baseline reset — recalibrating...")

    reader.release()
    cv2.destroyAllWindows()
    if face_executor:
        face_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()