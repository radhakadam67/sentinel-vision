"""
models/ensemble.py
Context-aware ensemble fusion layer.

What makes this novel:
  1. Severity scoring   — LOW / MEDIUM / HIGH / CRITICAL (not just a number)
  2. Weapon override    — Any weapon instantly forces CRITICAL regardless of other scores
  3. Crowd escalation   — Tracks sudden crowd surges as a separate risk signal
  4. Per-track cooldown — Avoids alert spam while still catching distinct events
"""

import time
from configs.settings import (
    WEIGHTS, ENSEMBLE_ALERT_THRESHOLD, ALERT_COOLDOWN_SECONDS,
    SEVERITY_LEVELS, CROWD_HIGH_DENSITY, CROWD_SURGE_FRAMES
)


def _get_severity(risk_score: float) -> str:
    """Map a 0-1 risk score to a human-readable severity label."""
    for label, (low, high) in SEVERITY_LEVELS.items():
        if low <= risk_score < high:
            return label
    return "LOW"


class EnsembleFusion:
    """
    Fuses scores from:
      - YOLOv8        (person presence + crowd density)
      - ByteTrack     (loitering score)
      - ActionClassifier (heuristic action recognition)
      - Autoencoder   (anomaly score)
      - WeaponDetector (weapon presence — CRITICAL override)

    Outputs a severity-scored risk assessment and fires an alert if warranted.
    Includes per-track cooldown to avoid alert spam.
    """

    def __init__(self):
        self.last_alert_time: dict[str, float] = {}
        self._person_count_history: list[int]  = []   # used for crowd surge detection
        print("[Ensemble] Fusion layer ready")

    def _cooldown_key(self, tracked_persons: list[dict], action: dict) -> str:
        track_ids = tuple(sorted(p["track_id"] for p in tracked_persons))
        return f"{track_ids}_{action.get('action', 'anomaly')}"

    def _in_cooldown(self, key: str) -> bool:
        last = self.last_alert_time.get(key, 0)
        return (time.time() - last) < ALERT_COOLDOWN_SECONDS

    def _crowd_escalation_score(self, person_count: int) -> tuple[float, list[str]]:
        """
        Detect two crowd risk signals:
          1. High density  — too many people in frame at once
          2. Sudden surge  — count jumped drastically in recent frames
        """
        reasons = []

        # Keep a rolling history of person counts
        self._person_count_history.append(person_count)
        if len(self._person_count_history) > CROWD_SURGE_FRAMES:
            self._person_count_history.pop(0)

        score = 0.0

        # Signal 1: high density
        if person_count >= CROWD_HIGH_DENSITY:
            score = max(score, 0.4)
            reasons.append(f"High crowd density: {person_count} people in frame")

        # Signal 2: sudden surge
        if len(self._person_count_history) >= CROWD_SURGE_FRAMES:
            baseline = self._person_count_history[0]
            surge    = person_count - baseline
            if surge >= 4:                          # 4+ new people appearing suddenly
                score = max(score, 0.6)
                reasons.append(f"Crowd surge detected: +{surge} people in {CROWD_SURGE_FRAMES} frames")

        return round(score, 3), reasons

    def fuse(
        self,
        tracked_persons: list[dict],
        action_result:   dict,
        anomaly_result:  dict,
        weapon_result:   dict | None = None,
    ) -> dict:
        """
        Compute ensemble risk score, severity level, and alert decision.

        Args:
            tracked_persons: Output from PersonTracker.update()
            action_result:   Output from ActionClassifier.classify()
            anomaly_result:  Output from AnomalyDetector.score()
            weapon_result:   Output from WeaponDetector.detect() (optional)

        Returns:
            {
                "risk_score":       float (0-1),
                "severity":         str   ("LOW" / "MEDIUM" / "HIGH" / "CRITICAL"),
                "should_alert":     bool,
                "alert_reasons":    list[str],
                "action":           str,
                "anomaly_score":    float,
                "person_count":     int,
                "weapon_detected":  bool,
                "weapons":          list[str],
                "weapon_boxes":     list,
                "crowd_score":      float,
            }
        """
        person_count  = len(tracked_persons)
        yolo_score    = min(person_count / 5.0, 1.0)
        tracker_score = self._loitering_score(tracked_persons)
        action_score  = action_result.get("confidence", 0.0) if action_result.get("is_alert") else 0.0
        anomaly_score = anomaly_result.get("anomaly_score", 0.0)

        # Crowd escalation
        crowd_score, crowd_reasons = self._crowd_escalation_score(person_count)

        # Weighted fusion (base)
        risk_score = (
            WEIGHTS["yolo"]        * yolo_score    +
            WEIGHTS["tracker"]     * tracker_score +
            WEIGHTS["slowfast"]    * action_score  +
            WEIGHTS["autoencoder"] * anomaly_score
        )

        # Boost for crowd escalation
        risk_score = min(risk_score + crowd_score * 0.3, 1.0)
        risk_score = round(risk_score, 3)

        # Determine alert reasons
        alert_reasons = list(crowd_reasons)
        if action_result.get("is_alert"):
            alert_reasons.append(
                f"Action detected: {action_result['action']} ({action_result['confidence']:.0%})"
            )
        if anomaly_result.get("is_anomaly"):
            alert_reasons.append(f"Scene anomaly score: {anomaly_score:.0%}")
        if any(p["loitering"] for p in tracked_persons):
            loiterers = [p for p in tracked_persons if p["loitering"]]
            alert_reasons.append(f"{len(loiterers)} person(s) loitering")

        # --- Weapon CRITICAL override ---
        weapon_detected = False
        weapons         = []
        weapon_boxes    = []

        if weapon_result and weapon_result.get("weapon_detected"):
            weapon_detected = True
            weapons         = weapon_result["weapons"]
            weapon_boxes    = weapon_result["weapon_boxes"]
            risk_score      = 1.0
            for w in set(weapons):
                alert_reasons.insert(0, f"⚠ WEAPON DETECTED: {w}")

        # Severity label
        severity = "CRITICAL" if weapon_detected else _get_severity(risk_score)

        # Alert decision
        should_alert = (risk_score >= ENSEMBLE_ALERT_THRESHOLD) and bool(alert_reasons)

        # Cooldown
        if should_alert:
            key = self._cooldown_key(tracked_persons, action_result)
            if self._in_cooldown(key):
                should_alert = False
            else:
                self.last_alert_time[key] = time.time()

        return {
            "risk_score":      risk_score,
            "severity":        severity,
            "should_alert":    should_alert,
            "alert_reasons":   alert_reasons,
            "action":          action_result.get("action", "unknown"),
            "anomaly_score":   anomaly_score,
            "person_count":    person_count,
            "weapon_detected": weapon_detected,
            "weapons":         weapons,
            "weapon_boxes":    weapon_boxes,
            "crowd_score":     crowd_score,
        }

    def _loitering_score(self, tracked_persons: list[dict]) -> float:
        if not tracked_persons:
            return 0.0
        loitering_count = sum(1 for p in tracked_persons if p["loitering"])
        return min(loitering_count / max(len(tracked_persons), 1), 1.0)
