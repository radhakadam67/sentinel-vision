# watchlist/utils/match_alert.py
# Fires alerts when a wanted person is matched in live video
# Includes camera location so police know exactly where to go

import os
import cv2
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MatchAlertDispatcher:
    """
    Sends high-priority alerts when a watchlist face match is found.
    Includes exact camera location so officers can respond immediately.
    """

    def __init__(self):
        self.twilio_sid    = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token  = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from   = os.getenv("TWILIO_FROM_NUMBER")
        self.officer_phone = os.getenv("OFFICER_PHONE")
        self.webhook_url   = os.getenv("ALERT_WEBHOOK_URL")

    def dispatch(self, match: dict, frame) -> str:
        """
        Fire all alert channels for a watchlist match.

        Args:
            match: Match result from FaceWatchlist.scan_frame()
            frame: Current video frame (for snapshot)

        Returns:
            Path to saved snapshot
        """
        snapshot_path = self._save_snapshot(frame, match)
        self._console_alert(match, snapshot_path)
        self._sms_alert(match)
        self._webhook_alert(match, snapshot_path)
        return snapshot_path

    def _save_snapshot(self, frame, match: dict) -> str:
        """Save annotated snapshot with face highlighted."""
        os.makedirs("watchlist/snapshots", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"watchlist/snapshots/match_{match['name'].replace(' ', '_')}_{ts}.jpg"

        annotated = frame.copy()
        top, right, bottom, left = match["face_bbox"]

        # Red box around matched face
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 0, 255), 3)
        cv2.putText(annotated, f"MATCH: {match['name']}", (left, top - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated, f"{match['confidence']:.1f}% | {match['location']}", (left, bottom + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)

        cv2.imwrite(path, annotated)
        return path

    def _console_alert(self, match: dict, snapshot_path: str):
        print("\n" + "!" * 60)
        print(f"  WATCHLIST MATCH FOUND")
        print(f"  Name      : {match['name']}")
        print(f"  Case ID   : {match['case_id'] or 'N/A'}")
        print(f"  Location  : {match['location']}")
        print(f"  Camera    : {match['camera_id']}")
        print(f"  Confidence: {match['confidence']:.1f}%")
        print(f"  Time      : {match['timestamp']}")
        print(f"  Snapshot  : {snapshot_path}")
        print("!" * 60 + "\n")

    def _sms_alert(self, match: dict):
        if not all([self.twilio_sid, self.twilio_token, self.twilio_from, self.officer_phone]):
            return
        try:
            from twilio.rest import Client
            client = Client(self.twilio_sid, self.twilio_token)
            body = (
                f"WANTED PERSON SPOTTED\n"
                f"Name: {match['name']}\n"
                f"Case: {match['case_id'] or 'N/A'}\n"
                f"Location: {match['location']}\n"
                f"Camera: {match['camera_id']}\n"
                f"Confidence: {match['confidence']:.1f}%\n"
                f"Time: {match['timestamp']}\n"
                f"RESPOND IMMEDIATELY"
            )
            client.messages.create(body=body, from_=self.twilio_from, to=self.officer_phone)
            print(f"[MatchAlert] SMS sent to {self.officer_phone}")
        except Exception as e:
            print(f"[MatchAlert] SMS failed: {e}")

    def _webhook_alert(self, match: dict, snapshot_path: str):
        if not self.webhook_url:
            return
        try:
            payload = {**match, "snapshot": snapshot_path, "alert_type": "WATCHLIST_MATCH"}
            requests.post(self.webhook_url, json=payload, timeout=5)
            print("[MatchAlert] Webhook sent")
        except Exception as e:
            print(f"[MatchAlert] Webhook failed: {e}")
