# Surveillance AI — Ensemble Model Pipeline

Real-time surveillance system using four ML models fused into one alert pipeline.

## Project Structure

```
surveillance_ai/
├── main.py                  # Entry point — run this
├── requirements.txt
├── .env.example             # Copy to .env and fill in your values
├── configs/
│   └── settings.py          # All thresholds, weights, model paths
├── models/
│   ├── detector.py          # YOLOv8 — person/object detection (COCO + VIRAT)
│   ├── tracker.py           # ByteTrack — multi-person tracking + loitering
│   ├── action_classifier.py # SlowFast — action classification (Kinetics-700 + UCF-Crime)
│   ├── anomaly_detector.py  # Autoencoder — scene anomaly (ShanghaiTech)
│   └── ensemble.py          # Weighted fusion of all four models
└── utils/
    └── alert.py             # Console + webhook + SMS alert dispatcher
```

## Datasets Used

| Model | Dataset | Download |
|---|---|---|
| YOLOv8 | COCO + VIRAT | https://cocodataset.org / https://viratdata.org |
| ByteTrack | MOT17 | https://motchallenge.net |
| SlowFast | Kinetics-700 + UCF-Crime | https://deepmind.com/research/open-source/kinetics |
| Autoencoder | ShanghaiTech | https://svip-lab.github.io/dataset/campus_dataset.html |

## Setup in VSCode

### 1. Open the project
```
File > Open Folder > select surveillance_ai/
```

### 2. Create a virtual environment (in VSCode terminal)
```bash
python -m venv venv
```

### 3. Activate it
```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your Twilio credentials and camera URL
```

### 6. Download YOLOv8 model (auto-downloads on first run)
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
mv yolov8m.pt models/
```

### 7. Train the autoencoder on ShanghaiTech (required before running)
```bash
# Download ShanghaiTech dataset first, then:
python -c "
from models.anomaly_detector import train_autoencoder
train_autoencoder('data/shanghaitech/train/frames', epochs=50)
"
```

### 8. Run the pipeline
```bash
python main.py
```

For webcam testing (no real camera needed):
- Open `configs/settings.py`
- Set `STREAM_URL = "0"` for your webcam

## Tuning Alerts

Edit `configs/settings.py`:

- `ENSEMBLE_ALERT_THRESHOLD` — raise to reduce false alarms, lower to catch more
- `LOITER_SECONDS` — how long before someone is flagged as loitering
- `WEIGHTS` — adjust which model contributes most to the risk score
- `ALERT_COOLDOWN_SECONDS` — minimum gap between repeated alerts

## VSCode Extensions Recommended

- Python (Microsoft)
- Pylance
- Python Debugger

## Hardware Requirements

- NVIDIA GPU with 8GB+ VRAM for real-time inference
- CPU-only works but will be slow (5-10 FPS)
- For cloud: AWS g4dn.xlarge (~$0.50/hr)
