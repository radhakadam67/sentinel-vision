# watchlist/watchlist_scanner.py
# Run this to start the city-wide wanted person scanner
#
# Usage:
#   python watchlist/watchlist_scanner.py
#
# Before running:
#   pip install face-recognition cmake dlib
#
# To add a wanted person first:
#   python watchlist/add_person.py

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_watchlist import FaceWatchlist
from match_alert import MatchAlertDispatcher

# --- Camera config ---
# For multiple cameras, duplicate this and run one process per camera
CAMERA_SOURCE = 0                              # 0 = webcam, or RTSP URL string
CAMERA_ID     = "CAM-001"
CAMERA_LOCATION = "High Street, City Centre"   # Physical location shown in alert

# Scan every N frames (reduce CPU load — faces don't move that fast)
SCAN_EVERY_N_FRAMES = 5

# Cooldown per person — don't alert twice for same match within this many seconds
MATCH_COOLDOWN = 120


def draw_watchlist_overlay(frame, matches: list[dict], watchlist_count: int, fps: float):
    """Draw face boxes and match banners on frame."""
    h, w = frame.shape[:2]

    # Status bar (top)
    cv2.rectangle(frame, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(frame, f"Watchlist: {watchlist_count} person(s) | Camera: {CAMERA_ID} | {CAMERA_LOCATION}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 90, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    for match in matches:
        top, right, bottom, left = match["face_bbox"]

        # Red box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)

        # Name tag below face box
        label = f"{match['name']}  {match['confidence']:.0f}%"
        cv2.rectangle(frame, (left, bottom), (right, bottom + 28), (0, 0, 200), -1)
        cv2.putText(frame, label, (left + 4, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Full screen alert banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 40), (w, 110), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Flash border
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        cv2.putText(frame, "WANTED PERSON DETECTED", (14, 68),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"{match['name']}  |  Case: {match['case_id'] or 'N/A'}  |  {match['confidence']:.0f}% match",
                    (14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    return frame


def main():
    print("\n=== Watchlist Face Scanner — Starting ===")

    watchlist = FaceWatchlist()
    alerter   = MatchAlertDispatcher()

    if len(watchlist.watchlist) == 0:
        print("\n[WARNING] Watchlist is empty.")
        print("  Add a wanted person first:")
        print("  python watchlist/add_person.py\n")

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera: {CAMERA_SOURCE}")
        return

    print(f"[Scanner] Running on {CAMERA_ID} — {CAMERA_LOCATION}")
    print("[Scanner] Press Q to quit\n")

    frame_count   = 0
    fps_timer     = time.time()
    fps_display   = 0.0
    last_match_time: dict[str, float] = {}  # name -> last alert time

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue

        frame_count += 1
        matches = []

        # Only scan every N frames to keep real-time performance
        if frame_count % SCAN_EVERY_N_FRAMES == 0:
            matches = watchlist.scan_frame(frame, CAMERA_ID, CAMERA_LOCATION)

            for match in matches:
                name = match["name"]
                last = last_match_time.get(name, 0)

                # Respect cooldown per person
                if time.time() - last >= MATCH_COOLDOWN:
                    last_match_time[name] = time.time()
                    alerter.dispatch(match, frame)

        # FPS
        if frame_count % 30 == 0:
            fps_display = 30 / max(time.time() - fps_timer, 0.001)
            fps_timer   = time.time()

        annotated = draw_watchlist_overlay(
            frame.copy(), matches, len(watchlist.watchlist), fps_display
        )

        cv2.imshow("Watchlist Scanner", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[Scanner] Stopped")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
