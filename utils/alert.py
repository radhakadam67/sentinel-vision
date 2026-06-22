# utils/alert.py
# Sends alerts to security officers when the ensemble triggers
# Supports: console print, email (Gmail), ntfy.sh push notification, webhook, SMS via Twilio

import os
import json
import time
import smtplib
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class AlertDispatcher:
    """
    Receives alert events from the ensemble and dispatches them.

    Channels supported:
      1. Console  — always on, useful for development
      2. Email    — Gmail via smtplib (built-in, no extra libraries)
      3. ntfy.sh  — free push notification (requires ntfy phone app)
      4. Webhook  — POST to any URL
      5. Twilio   — real SMS to security officer's phone
    """

    # ntfy priority levels (1=min … 5=max/urgent)
    _NTFY_PRIORITY = {
        "LOW":      "2",
        "MEDIUM":   "3",
        "HIGH":     "4",
        "CRITICAL": "5",   # max — phone will buzz even in Do Not Disturb
    }

    def __init__(self):
        self.twilio_sid    = os.getenv("TWILIO_ACCOUNT_SID")
        self.twilio_token  = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_from   = os.getenv("TWILIO_FROM_NUMBER")   # e.g. "+441234567890"
        self.officer_phone = os.getenv("OFFICER_PHONE")        # e.g. "+447700900000"
        self.webhook_url   = os.getenv("ALERT_WEBHOOK_URL")    # Your push server endpoint
        self.camera_name   = os.getenv("CAMERA_NAME", "Camera 1")

        # ntfy.sh — free push notifications, no account needed
        # Set NTFY_TOPIC to any secret string, e.g. "radha-surv-alerts-7734"
        # Then subscribe to that topic in the ntfy phone app
        self.ntfy_topic    = os.getenv("NTFY_TOPIC")           # e.g. "radha-surv-alerts-7734"
        self.ntfy_server   = os.getenv("NTFY_SERVER", "https://ntfy.sh")  # or self-hosted

        # Email (Gmail) — no extra libraries needed, uses Python built-in smtplib
        # Use a Gmail App Password (NOT your normal password):
        #   Google Account → Security → 2-Step Verification → App Passwords
        self.email_sender   = os.getenv("EMAIL_SENDER")         # your Gmail address
        self.email_password = os.getenv("EMAIL_APP_PASSWORD")   # 16-char App Password
        self.email_receiver = os.getenv("EMAIL_RECEIVER")       # where alerts go (can be same)
        self.email_smtp     = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
        self.email_port     = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self._last_email_time: dict = {}   # throttle: 1 email/min per severity

    def dispatch(self, fusion_result: dict, frame_path: str = None):
        """
        Send an alert through all configured channels.

        Args:
            fusion_result: Output from EnsembleFusion.fuse()
            frame_path:    Optional path to saved snapshot image
        """
        if not fusion_result.get("should_alert"):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reasons   = ", ".join(fusion_result["alert_reasons"])
        score     = fusion_result["risk_score"]
        severity  = fusion_result.get("severity", "MEDIUM")

        payload = {
            "camera":    self.camera_name,
            "timestamp": timestamp,
            "risk_score": score,
            "severity":  severity,
            "reasons":   fusion_result["alert_reasons"],
            "action":    fusion_result["action"],
            "persons":   fusion_result["person_count"],
            "weapons":   fusion_result.get("weapons", []),
            "snapshot":  frame_path
        }

        self._console_alert(payload)
        self._email_alert(payload)         # email with snapshot attached
        self._ntfy_alert(payload)          # push notification to phone (needs app)
        self._webhook_alert(payload)
        self._sms_alert(timestamp, reasons, score, severity)

    def _console_alert(self, payload: dict):
        severity_icons = {
            "LOW":      "🟡",
            "MEDIUM":   "🟠",
            "HIGH":     "🔴",
            "CRITICAL": "🚨",
        }
        icon = severity_icons.get(payload["severity"], "⚠")
        print("\n" + "=" * 60)
        print(f"  {icon} {payload['severity']} ALERT — {payload['camera']} — {payload['timestamp']}")
        print(f"  Risk score : {payload['risk_score']:.0%}")
        if payload["weapons"]:
            print(f"  ⚠ WEAPONS  : {', '.join(payload['weapons'])}")
        print(f"  Reasons    : {', '.join(payload['reasons'])}")
        print(f"  Action     : {payload['action']}")
        print(f"  Persons    : {payload['persons']}")
        if payload["snapshot"]:
            print(f"  Snapshot   : {payload['snapshot']}")
        print("=" * 60 + "\n")

    def _email_alert(self, payload: dict):
        """Send email alert with snapshot attached using Python built-in smtplib."""
        if not all([self.email_sender, self.email_password, self.email_receiver]):
            return

        # Throttle: max 1 email per minute per severity level
        severity = payload["severity"]
        now = time.time()
        if now - self._last_email_time.get(severity, 0) < 60:
            return
        self._last_email_time[severity] = now

        severity_emoji = {"LOW": "🟡", "MEDIUM": "🟠", "HIGH": "🔴", "CRITICAL": "🚨"}
        icon = severity_emoji.get(severity, "⚠️")
        subject = f"{icon} {severity} ALERT — {payload['camera']} — {payload['timestamp']}"

        weapons_html = (
            f"<p style='color:#cc0000;font-weight:bold'>⚠️ WEAPONS DETECTED: "
            f"{', '.join(payload['weapons'])}</p>"
            if payload["weapons"] else ""
        )

        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;background:#111;color:#eee;padding:20px">
          <h2 style="color:#ff4444">{icon} {severity} SURVEILLANCE ALERT</h2>
          <table style="border-collapse:collapse;width:100%">
            <tr><td style="padding:6px;color:#aaa">📷 Camera</td>
                <td style="padding:6px"><b>{payload['camera']}</b></td></tr>
            <tr><td style="padding:6px;color:#aaa">🕐 Time</td>
                <td style="padding:6px">{payload['timestamp']}</td></tr>
            <tr><td style="padding:6px;color:#aaa">⚡ Risk Score</td>
                <td style="padding:6px"><b>{payload['risk_score']:.0%}</b></td></tr>
            <tr><td style="padding:6px;color:#aaa">🎬 Action</td>
                <td style="padding:6px">{payload['action']}</td></tr>
            <tr><td style="padding:6px;color:#aaa">👥 Persons</td>
                <td style="padding:6px">{payload['persons']}</td></tr>
            <tr><td style="padding:6px;color:#aaa">🔍 Reasons</td>
                <td style="padding:6px">{', '.join(payload['reasons'])}</td></tr>
          </table>
          {weapons_html}
          <p style="color:#888;margin-top:20px">Snapshot attached if available.</p>
        </body></html>
        """

        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = subject
            msg["From"]    = self.email_sender
            msg["To"]      = self.email_receiver
            msg.attach(MIMEText(html_body, "html"))

            # Attach snapshot if it exists
            snapshot = payload.get("snapshot")
            if snapshot and os.path.isfile(snapshot):
                with open(snapshot, "rb") as img_file:
                    img = MIMEImage(img_file.read(), name=os.path.basename(snapshot))
                    img.add_header("Content-Disposition", "attachment",
                                   filename=os.path.basename(snapshot))
                    msg.attach(img)

            with smtplib.SMTP(self.email_smtp, self.email_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.email_sender, self.email_password)
                server.sendmail(self.email_sender, self.email_receiver, msg.as_string())

            print(f"[Alert] Email sent → {self.email_receiver}")

        except Exception as e:
            print(f"[Alert] Email failed: {e}")

    def _ntfy_alert(self, payload: dict):
        """Send instant push notification via ntfy.sh (free, no account needed)."""
        if not self.ntfy_topic:
            return
        severity = payload["severity"]
        priority = self._NTFY_PRIORITY.get(severity, "3")
        weapons_line = f" | WEAPONS: {', '.join(payload['weapons'])}" if payload["weapons"] else ""
        message = (
            f"📍 {payload['camera']} | {payload['timestamp']}\n"
            f"Risk: {payload['risk_score']:.0%}{weapons_line}\n"
            f"{', '.join(payload['reasons'])}\n"
            f"Persons: {payload['persons']}"
        )
        try:
            resp = requests.post(
                f"{self.ntfy_server}/{self.ntfy_topic}",
                data=message.encode("utf-8"),
                headers={
                    "Title":    f"🚨 {severity} ALERT — {payload['camera']}",
                    "Priority": priority,
                    "Tags":     "rotating_light,camera,warning",
                },
                timeout=5,
            )
            if resp.status_code == 200:
                print(f"[Alert] ntfy push sent → {self.ntfy_server}/{self.ntfy_topic}")
            else:
                print(f"[Alert] ntfy returned {resp.status_code}")
        except Exception as e:
            print(f"[Alert] ntfy failed: {e}")

    def _webhook_alert(self, payload: dict):
        if not self.webhook_url:
            return
        try:
            r = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            if r.status_code == 200:
                print(f"[Alert] Webhook sent OK")
            else:
                print(f"[Alert] Webhook returned {r.status_code}")
        except Exception as e:
            print(f"[Alert] Webhook failed: {e}")

    def _sms_alert(self, timestamp: str, reasons: str, score: float, severity: str = "MEDIUM"):
        if not all([self.twilio_sid, self.twilio_token, self.twilio_from, self.officer_phone]):
            return
        try:
            from twilio.rest import Client
            client = Client(self.twilio_sid, self.twilio_token)
            body = (
                f"{severity} ALERT | {self.camera_name} | {timestamp}\n"
                f"Risk: {score:.0%} | {reasons}\n"
                f"View live feed immediately."
            )
            client.messages.create(
                body=body,
                from_=self.twilio_from,
                to=self.officer_phone
            )
            print(f"[Alert] SMS sent to {self.officer_phone}")
        except Exception as e:
            print(f"[Alert] SMS failed: {e}")
