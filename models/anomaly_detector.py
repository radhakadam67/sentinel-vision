# models/anomaly_detector.py
# Convolutional Autoencoder for scene anomaly detection
#
# How it works:
#   1. Train ONLY on normal footage — the model learns to reconstruct normal scenes well
#   2. At inference, reconstruct each frame and measure the error
#   3. High reconstruction error = the scene looks unusual = anomaly alert
#
# Training dataset:
#   ShanghaiTech Campus dataset — 437 surveillance scenes, 130 anomaly events
#   Download: https://svip-lab.github.io/dataset/campus_dataset.html

import torch
import torch.nn as nn
import numpy as np
import cv2
from configs.settings import (
    AUTOENCODER_MODEL_PATH, ANOMALY_THRESHOLD, ANOMALY_WARMUP_FRAMES
)


class ConvAutoencoder(nn.Module):
    """
    Lightweight convolutional autoencoder.
    Encoder compresses the frame, decoder reconstructs it.
    High reconstruction loss = anomaly.
    """

    def __init__(self):
        super().__init__()

        # Encoder: shrinks 1x64x64 → 16x8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 8x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 8x32x32
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16x16
            nn.Conv2d(16, 16, kernel_size=3, padding=1), # 16x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x8x8
        )

        # Decoder: expands 16x8x8 → 1x64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),  # 16x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),   # 8x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),    # 1x64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AnomalyDetector:
    """
    Loads trained autoencoder and scores each frame for anomalies.

    Adaptive baseline:
      During the first ANOMALY_WARMUP_FRAMES frames the detector collects
      the reconstruction-loss distribution of YOUR camera's normal scene.
      It then sets a data-driven threshold = mean + 2*std of those losses.
      This prevents false-positive anomalies when running with untrained
      (random-weight) weights, or on cameras with unusual lighting.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = ConvAutoencoder().to(self.device)
        self.weights_loaded = self._load_weights()
        self.model.eval()
        self.loss_fn = nn.MSELoss()

        # Adaptive baseline state
        self._warmup_losses: list[float] = []
        self._dynamic_threshold: float | None = None   # set after warmup
        self._calibrated = False

        src = "loaded" if self.weights_loaded else "random (calibrating…)"
        print(f"[AnomalyDetector] Ready on {self.device} — weights: {src}")

    def _load_weights(self) -> bool:
        """Returns True if weights were loaded from disk, False if random."""
        try:
            checkpoint = torch.load(
                AUTOENCODER_MODEL_PATH,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(checkpoint)
            print("[AnomalyDetector] Weights loaded successfully")
            return True
        except Exception as e:
            print(f"[AnomalyDetector] No saved weights ({e}) — adaptive baseline active")
            return False

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Resize to 64x64, convert to grayscale, normalise to [0,1]."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)  # 1x1x64x64
        return tensor.to(self.device)

    def score(self, frame: np.ndarray) -> dict:
        """
        Score a single frame for anomaly.

        During the warmup window the detector is calibrating and will
        never return is_anomaly=True (avoids floods of false alerts).

        Returns:
            {
                "anomaly_score": float (0-1),
                "is_anomaly":    bool,
                "calibrating":   bool   # True during warmup
            }
        """
        with torch.no_grad():
            tensor = self._preprocess(frame)
            reconstructed = self.model(tensor)
            loss = self.loss_fn(reconstructed, tensor).item()

        # --- Adaptive baseline accumulation ---
        if not self._calibrated:
            self._warmup_losses.append(loss)
            if len(self._warmup_losses) >= ANOMALY_WARMUP_FRAMES:
                mu  = float(np.mean(self._warmup_losses))
                std = float(np.std(self._warmup_losses))
                # dynamic threshold: mean + 2 std deviations (clips to config max)
                self._dynamic_threshold = min(mu + 2.0 * std, ANOMALY_THRESHOLD * 0.1)
                self._calibrated = True
                print(
                    f"[AnomalyDetector] Baseline calibrated — "
                    f"μ={mu:.5f}, σ={std:.5f}, "
                    f"threshold={self._dynamic_threshold:.5f}"
                )
            # Still warming up
            score = min(loss * 10.0, 1.0)
            return {"anomaly_score": round(score, 3), "is_anomaly": False, "calibrating": True}

        # --- Normal scoring (post-calibration) ---
        threshold = self._dynamic_threshold if self._dynamic_threshold else (ANOMALY_THRESHOLD * 0.1)
        score = min(loss / (threshold * 1.5 + 1e-8), 1.0)  # normalize relative to baseline

        return {
            "anomaly_score": round(score, 3),
            "is_anomaly":    score >= ANOMALY_THRESHOLD,
            "calibrating":   False,
        }


def train_autoencoder(data_dir: str, epochs: int = 50, save_path: str = "models/autoencoder.pt"):
    """
    Train the autoencoder on normal footage only (ShanghaiTech train split).

    Args:
        data_dir:  Path to folder of normal frame images (.jpg/.png)
        epochs:    Training epochs
        save_path: Where to save the trained model

    Usage:
        python -c "from models.anomaly_detector import train_autoencoder; train_autoencoder('data/shanghaitech/train/frames')"
    """
    import os
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    class FrameDataset(Dataset):
        def __init__(self, folder):
            self.paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".png"))
            ]
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])

        def __len__(self): return len(self.paths)

        def __getitem__(self, idx):
            img = cv2.imread(self.paths[idx])
            return self.transform(img)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = ConvAutoencoder().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader  = DataLoader(FrameDataset(data_dir), batch_size=32, shuffle=True)

    print(f"[Training] Starting on {len(loader.dataset)} frames for {epochs} epochs")

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs} — loss: {avg:.5f}")

    torch.save(model.state_dict(), save_path)
    print(f"[Training] Model saved to {save_path}")
