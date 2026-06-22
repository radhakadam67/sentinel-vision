import cv2
import os
import time
import sys

# Ensure we can import from the models directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.anomaly_detector import train_autoencoder

def collect_and_train():
    data_dir = "data/webcam/train/frames"
    os.makedirs(data_dir, exist_ok=True)
    
    print("\n--- Phase 1: Recording Normal Footage ---")
    print("Please act normal in front of the camera for the next 10 seconds.")
    print("The system is memorizing the room...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found! Cannot record training data.")
        return

    frames_captured = 0
    for i in range(100):
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{data_dir}/frame_{i}.jpg", frame)
            frames_captured += 1
        time.sleep(0.1)  # Space out the frames a bit
        
    cap.release()
    print(f"Success! {frames_captured} training frames gathered.")
    
    print("\n--- Phase 2: Training the Neural Network ---")
    print("Training the Autoencoder on the recorded footage...")
    # Train for 25 epochs so it finishes relatively quickly but still learns the basics
    train_autoencoder(data_dir=data_dir, epochs=25, save_path="models/autoencoder.pt")
    
    print("\n[FINISHED] Autoencoder 'brain' (models/autoencoder.pt) has been generated!")
    print("The pipeline will now be able to detect anomalies correctly.")

if __name__ == "__main__":
    collect_and_train()
