"""
test_email_alert.py
───────────────────
Quick test to verify your email alert is working.
Run this BEFORE starting the full surveillance pipeline.

Usage:
    python test_email_alert.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.alert import AlertDispatcher

def main():
    print("\n=== Email Alert Test ===\n")

    alerter = AlertDispatcher()

    # Check credentials are loaded
    if not alerter.email_sender:
        print("❌  EMAIL_SENDER not set in .env")
        print("    Open .env and fill in your Gmail address.\n")
        sys.exit(1)
    if not alerter.email_password:
        print("❌  EMAIL_APP_PASSWORD not set in .env")
        print("    Go to myaccount.google.com → Security → App Passwords\n")
        sys.exit(1)
    if not alerter.email_receiver:
        print("❌  EMAIL_RECEIVER not set in .env\n")
        sys.exit(1)

    print(f"📧  Sending test alert FROM : {alerter.email_sender}")
    print(f"📬  Sending test alert TO   : {alerter.email_receiver}")
    print(f"📷  Camera name             : {alerter.camera_name}\n")

    # Build a fake fusion_result that looks exactly like what the real pipeline sends
    fake_result = {
        "should_alert":   True,
        "risk_score":     0.91,
        "severity":       "HIGH",
        "alert_reasons":  ["Fighting detected", "Crowd density high"],
        "action":         "fighting",
        "person_count":   3,
        "weapons":        [],        # set to e.g. ["knife"] to test weapon line
        "anomaly_score":  0.78,
        "weapon_boxes":   [],
    }

    print("⏳  Sending email now...\n")
    alerter.dispatch(fake_result, frame_path=None)   # no snapshot for the test
    print("\n✅  Done! Check your inbox (and spam folder just in case).")
    print("    If you got the email, your alert system is fully working!\n")

if __name__ == "__main__":
    main()
