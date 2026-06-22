# models/face_watchlist.py
# Facial recognition watchlist system
#
# How it works:
#   1. Police upload a photo of the wanted person
#   2. System extracts a 128-dimension face embedding from the photo
#   3. Every camera frame is scanned for faces in real time
#   4. Each detected face is compared against all watchlist embeddings
#   5. If similarity is above threshold — MATCH FOUND — alert fires with location
#
# Library: face_recognition (built on dlib)
# Install: pip install face-recognition
#
# Dataset used for the underlying model:
#   - LFW (Labeled Faces in the Wild) — 13,000 face images
#   - face_recognition library is pre-trained — no training needed

import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import face_recognition
    FACE_LIB_AVAILABLE = True
except ImportError:
    FACE_LIB_AVAILABLE = False
    print("[Watchlist] WARNING: face_recognition not installed")
    print("  Run: pip install face-recognition")

WATCHLIST_DB = "watchlist/watchlist.pkl"   # Saved embeddings database
MATCH_THRESHOLD = 0.55                      # Lower = stricter match (0.4 strict, 0.6 lenient)


class FaceWatchlist:
    """
    Manages the wanted persons database and matches faces from live video.

    Each entry in the watchlist stores:
      - name:      person's name or case ID
      - embedding: 128-d face vector extracted from uploaded photo
      - photo_path: path to original uploaded photo
      - added_at:  timestamp when uploaded
      - 
      

    def __init__(self):
        self.watchlist: list[dict] = []
        self._load_watchlist()
        print(f"[Watchlist] Loaded {len(self.watchlist)} wanted person(s)")

    def _load_watchlist(self):
        """Load saved watchlist from disk."""
        if os.path.exists(WATCHLIST_DB):
            with open(WATCHLIST_DB, "rb") as f:
                self.watchlist = pickle.load(f)

    def _save_watchlist(self):
        """Persist watchlist to disk."""
        os.makedirs(os.path.dirname(WATCHLIST_DB), exist_ok=True)
        with open(WATCHLIST_DB, "wb") as f:
            pickle.dump(self.watchlist, f)

    def add_person(self, photo_path: str, name: str, case_id: str = "") -> dict:
        """
        Add a wanted person from an uploaded photo.

        Args:
            photo_path: Path to uploaded photo (.jpg / .png)
            name:       Person name or alias
            case_id:    Police case reference number

        Returns:
            Result dict with success status and message
        """
        if not FACE_LIB_AVAILABLE:
            return {"success": False, "message": "face_recognition library not installed"}

        image = face_recognition.load_image_file(photo_path)
        embeddings = face_recognition.face_encodings(image)

        if len(embeddings) == 0:
            return {"success": False, "message": "No face detected in the uploaded photo. Use a clear front-facing image."}

        if len(embeddings) > 1:
            return {"success": False, "message": f"Multiple faces found ({len(embeddings)}). Upload a photo with only one person."}

        entry = {
            "name":       name,
            "case_id":    case_id,
            "embedding":  embeddings[0],
            "photo_path": photo_path,
            "added_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.watchlist.append(entry)
        self._save_watchlist()

        print(f"[Watchlist] Added: {name} (Case: {case_id or 'N/A'})")
        return {"success": True, "message": f"{name} added to watchlist successfully"}

    def remove_person(self, name: str) -> bool:
        """Remove a person from the watchlist by name."""
        original_count = len(self.watchlist)
        self.watchlist = [p for p in self.watchlist if p["name"] != name]
        self._save_watchlist()
        return len(self.watchlist) < original_count

    def scan_frame(self, frame, camera_id: str = "Unknown", location: str = "Unknown") -> list[dict]:
        """
        Scan a single video frame against the entire watchlist.

        Args:
            frame:      BGR frame from OpenCV
            camera_id:  Which camera this frame is from
            location:   Physical location of the camera (e.g. "High Street, Manchester")

        Returns:
            List of matches found, each as:
            {
                "name":       str,
                "case_id":    str,
                "confidence": float,
                "location":   str,
                "camera_id":  str,
                "timestamp":  str,
                "face_bbox":  [top, right, bottom, left]
            }
        """
        if not FACE_LIB_AVAILABLE or not self.watchlist:
            return []

        # Convert BGR (OpenCV) to RGB (face_recognition expects RGB)
        rgb_frame = frame[:, :, ::-1]

        # Detect all faces in this frame
        face_locations  = face_recognition.face_locations(rgb_frame, model="hog")
        face_embeddings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_embeddings:
            return []

        matches = []
        watchlist_embeddings = [p["embedding"] for p in self.watchlist]

        for face_embedding, face_location in zip(face_embeddings, face_locations):
            # Compare this face against all watchlist entries at once
            distances = face_recognition.face_distance(watchlist_embeddings, face_embedding)
            best_idx  = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist <= MATCH_THRESHOLD:
                person     = self.watchlist[best_idx]
                confidence = round((1 - best_dist) * 100, 1)

                match = {
                    "name":       person["name"],
                    "case_id":    person["case_id"],
                    "confidence": confidence,
                    "location":   location,
                    "camera_id":  camera_id,
                    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "face_bbox":  list(face_location)   # top, right, bottom, left
                }
                matches.append(match)
                print(f"[MATCH] {person['name']} spotted at {location} — {confidence:.1f}% confidence")

        return matches

    def list_watchlist(self) -> list[dict]:
        """Return all watchlist entries (without embeddings for display)."""
        return [
            {
                "name":      p["name"],
                "case_id":   p["case_id"],
                "added_at":  p["added_at"],
                "photo_path": p["photo_path"]
            }
            for p in self.watchlist
        ]
