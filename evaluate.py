# evaluate.py
# Tests accuracy of each model in the pipeline
# Run: python evaluate.py
#
# What it measures:
#   - Detection: how many real people are found vs missed
#   - False alarms: how often it alerts when nothing is happening
#   - Loitering: does it trigger at the right time

import cv2
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import ObjectDetector
from models.tracker import PersonTracker
from models.anomaly_detector import AnomalyDetector
from models.ensemble import EnsembleFusion


def test_webcam_detection():
    """
    Live accuracy test — shows stats in real time.
    Stand in front of camera, walk around, then step out.
    Watch the numbers update live.

    Press Q to stop and see final summary.
    """
    print("\n=== Accuracy Evaluation ===")
    print("Stand in front of camera to test detection.")
    print("Press Q to quit and see results.\n")

    detector = ObjectDetector()
    tracker  = PersonTracker()
    anomaly  = AnomalyDetector()
    ensemble = EnsembleFusion()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible — grant camera permission first")
        return

    # Counters
    total_frames      = 0
    frames_with_person = 0
    total_alerts      = 0
    false_alert_frames = 0   # Alerts when you manually mark scene as empty
    detection_times   = []

    print("Controls:")
    print("  Q     = quit and show results")
    print("  SPACE = mark current frame as EMPTY (no person) — for false alarm testing\n")

    scene_is_empty = False   # You toggle this with spacebar

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        start = time.time()

        detections      = detector.detect(frame)
        tracked         = tracker.update(detections, frame.shape[:2])
        anomaly_result  = anomaly.score(frame)
        fusion          = ensemble.fuse(tracked, {"action": "normal", "action_id": 0,
                                                   "confidence": 0.0, "is_alert": False},
                                        anomaly_result)

        elapsed = (time.time() - start) * 1000
        detection_times.append(elapsed)

        person_count = len(tracked)
        if person_count > 0:
            frames_with_person += 1

        if fusion["should_alert"]:
            total_alerts += 1
            if scene_is_empty:
                false_alert_frames += 1

        # Draw stats overlay
        h, w = frame.shape[:2]
        avg_ms = sum(detection_times[-30:]) / min(len(detection_times), 30)

        cv2.putText(frame, f"Persons detected: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, f"Inference time: {avg_ms:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.putText(frame, f"Risk score: {fusion['risk_score']:.0%}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        cv2.putText(frame, f"Total alerts: {total_alerts}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)
        cv2.putText(frame, f"False alerts: {false_alert_frames}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)

        # Scene empty indicator
        label = "SCENE: EMPTY (space)" if scene_is_empty else "SCENE: OCCUPIED"
        color = (0, 0, 200) if scene_is_empty else (0, 200, 0)
        cv2.putText(frame, label, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw bounding boxes
        for person in tracked:
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, f"ID:{person['track_id']}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        cv2.imshow("Accuracy Evaluation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            scene_is_empty = not scene_is_empty
            print(f"  Scene marked as: {'EMPTY' if scene_is_empty else 'OCCUPIED'}")

    cap.release()
    cv2.destroyAllWindows()

    # --- Final report ---
    avg_inference = sum(detection_times) / max(len(detection_times), 1)
    detection_rate = (frames_with_person / max(total_frames, 1)) * 100
    false_alarm_rate = (false_alert_frames / max(total_alerts, 1)) * 100 if total_alerts > 0 else 0

    print("\n" + "=" * 50)
    print("  ACCURACY REPORT")
    print("=" * 50)
    print(f"  Total frames processed : {total_frames}")
    print(f"  Frames with person     : {frames_with_person} ({detection_rate:.1f}%)")
    print(f"  Avg inference time     : {avg_inference:.1f}ms per frame")
    print(f"  Estimated FPS          : {1000 / max(avg_inference, 1):.1f}")
    print(f"  Total alerts fired     : {total_alerts}")
    print(f"  False alerts           : {false_alert_frames} ({false_alarm_rate:.1f}% of alerts)")
    print("=" * 50)

    print("\nWhat these numbers mean:")
    if avg_inference < 100:
        print("  Inference: GOOD — fast enough for real-time")
    else:
        print("  Inference: SLOW — consider using yolov8n.pt (nano) for speed")

    if false_alarm_rate < 10:
        print("  False alarms: GOOD — under 10%")
    elif false_alarm_rate < 25:
        print("  False alarms: OK — raise ENSEMBLE_ALERT_THRESHOLD in settings.py")
    else:
        print("  False alarms: HIGH — raise ENSEMBLE_ALERT_THRESHOLD in settings.py")


if __name__ == "__main__":
    test_webcam_detection()
