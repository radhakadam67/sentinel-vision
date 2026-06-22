"""
models/weapon_detector.py
Detects weapons (knives, bladed objects) in each frame using YOLOv8.

Why this matters:
  Detecting a weapon immediately upgrades the alert severity to CRITICAL
  and overrides the regular ensemble risk score to 1.0, ensuring an
  emergency dispatch response is triggered no matter what the other models say.

COCO proxy classes used:
  43 = knife  (directly useful)
  76 = scissors (proxy for bladed objects)

For a real deployment, fine-tune on a dedicated weapons dataset such as:
  - Open Images V7 (Knife, Handgun, Shotgun labels available)
  - COCO-Weapons extensions
"""

import numpy as np
from ultralytics import YOLO
from configs.settings import WEAPON_MODEL_PATH, WEAPON_CLASSES, WEAPON_CONFIDENCE

WEAPON_LABEL_MAP = {
    43: "KNIFE",
    76: "BLADED OBJECT",
}


class WeaponDetector:
    def __init__(self):
        self.model = YOLO(WEAPON_MODEL_PATH)
        self.confidence = WEAPON_CONFIDENCE
        self.classes = WEAPON_CLASSES
        print(f"[WeaponDetector] Loaded — scanning for: {list(WEAPON_LABEL_MAP.values())}")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Scan a single frame for weapons.

        Returns:
            {
                "weapon_detected": bool,
                "weapons":         list[str],   e.g. ["KNIFE", "KNIFE"]
                "weapon_boxes":    list[list],  bounding boxes [[x1,y1,x2,y2], ...]
                "weapon_score":    float,        1.0 if any weapon found, else 0.0
            }
        """
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            verbose=False
        )[0]

        weapons_found = []
        weapon_boxes  = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = WEAPON_LABEL_MAP.get(cls_id, "UNKNOWN WEAPON")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            weapons_found.append(label)
            weapon_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        detected = len(weapons_found) > 0

        if detected:
            print(f"[WeaponDetector] ⚠️  WEAPON DETECTED: {weapons_found}")

        return {
            "weapon_detected": detected,
            "weapons":         weapons_found,
            "weapon_boxes":    weapon_boxes,
            "weapon_score":    1.0 if detected else 0.0
        }
