# configs/settings.py
# Central config for all models and alert thresholds

# ---------------------------------------------------------------------------
# CAMERA SOURCE
#   0               → built-in / USB webcam
#   1, 2 …          → second / third camera
#   "rtsp://…"      → IP camera RTSP stream
#   "/path/to/file" → pre-recorded video file (for testing)
# ---------------------------------------------------------------------------
CAMERA_SOURCE = 0

# Legacy alias (kept so nothing else breaks)
STREAM_URL = "rtsp://your_camera_ip/stream"

FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
FPS_TARGET   = 30

# --- YOLOv8 (Object Detection) ---
YOLO_MODEL_PATH  = "yolov8n.pt"   # nano — fast enough for real-time on CPU
YOLO_CONFIDENCE  = 0.40           # slightly lower to catch partially visible people
YOLO_CLASSES     = [0]            # 0 = person only

# --- Weapon Detection (COCO class IDs) ---
WEAPON_MODEL_PATH  = "yolov8n.pt"
WEAPON_CLASSES     = [43, 76]     # 43=knife, 76=scissors (COCO proxies)
WEAPON_CONFIDENCE  = 0.38         # slightly more sensitive — better safe than sorry

# --- ByteTrack (Multi-Person Tracking) ---
TRACK_MAX_AGE  = 40   # frames to hold a lost track (higher = better re-ID)
TRACK_MIN_HITS = 2    # confirm a new track after 2 hits (faster than 3 for webcam)

# --- SlowFast (Action Classification) ---
SLOWFAST_MODEL_PATH = "models/slowfast_r50.pt"
SLOWFAST_WINDOW     = 32
SLOWFAST_CONFIDENCE = 0.55
ACTION_LABELS = {
    0: "normal",
    1: "loitering",
    2: "fighting",
    3: "running",
    4: "vandalism",
    5: "robbery",
}
ALERT_ACTIONS = [1, 2, 4, 5]

# --- Autoencoder (Anomaly Detection) ---
AUTOENCODER_MODEL_PATH = "models/autoencoder.pt"
ANOMALY_THRESHOLD      = 0.70   # raised — random-weight model causes false anomalies

# Adaptive baseline: calibrate on first N frames before enabling anomaly alerts
ANOMALY_WARMUP_FRAMES  = 120    # ~4 s at 30 fps — enough to learn background

# --- Ensemble Fusion Weights ---
# Must sum to 1.0
WEIGHTS = {
    "yolo":        0.10,   # presence alone is weak signal
    "tracker":     0.20,   # loitering is reliable
    "slowfast":    0.50,   # action flag is most decisive
    "autoencoder": 0.20,   # scene anomaly (auto-calibrated)
}
ENSEMBLE_ALERT_THRESHOLD = 0.52   # slightly above 0.5 to cut marginal noise

# --- Severity Scoring Thresholds ---
SEVERITY_LEVELS = {
    "LOW":      (0.52, 0.62),
    "MEDIUM":   (0.62, 0.76),
    "HIGH":     (0.76, 0.90),
    "CRITICAL": (0.90, 1.01),
}

# --- Crowd Escalation Risk ---
CROWD_HIGH_DENSITY = 5         # 5+ people in frame
CROWD_SURGE_FRAMES = 15        # surge window (frames)

# --- Loitering ---
# 20 s for indoor/close camera; raise to 45 for street cameras
LOITER_SECONDS = 20

# --- Alert Cooldown ---
ALERT_COOLDOWN_SECONDS = 20    # 20 s cooldown — shorter so distinct events aren't missed

# --- Performance ---
# Process every Nth frame for heavy models (weapon, pose) to keep real-time
HEAVY_MODEL_SKIP = 2   # run weapon + pose every 2 frames
