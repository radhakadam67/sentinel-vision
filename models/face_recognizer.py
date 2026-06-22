import torch
import cv2
import os
import numpy as np
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceWatchlist:
    def __init__(self, watchlist_dir="watchlist_faces"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("[FaceWatchlist] Loading MTCNN and InceptionResnetV1...")
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.watchlist_dir = watchlist_dir
        self.known_embeddings = {}
        
        os.makedirs(self.watchlist_dir, exist_ok=True)
        self._load_watchlist()

    def _load_watchlist(self):
        print(f"[FaceWatchlist] Scanning {self.watchlist_dir} for known faces...")
        for file in os.listdir(self.watchlist_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(file)[0]
                path = os.path.join(self.watchlist_dir, file)
                img = Image.open(path).convert('RGB')
                
                # Look for a .txt file with the exact same name for related info
                info_text = ""
                info_path = os.path.join(self.watchlist_dir, f"{base_name}.txt")
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        info_text = f.read().strip()
                
                # Format the name (replace underscores with spaces)
                display_name = base_name.replace("_", " ")
                if info_text:
                    display_name = f"{display_name} ({info_text})"
                
                img_cropped = self.mtcnn(img)
                if img_cropped is not None:
                    img_embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device))
                    self.known_embeddings[display_name] = img_embedding.detach().cpu()
                    print(f"  -> Added {display_name} to watchlist.")
                else:
                    print(f"  -> WARNING: No face found in {file}!")
                    
    def check_persons(self, frame: np.ndarray, tracked_persons: list) -> list:
        matches = []
        if not self.known_embeddings:
            return matches
            
        h, w = frame.shape[:2]
        for person in tracked_persons:
            x1, y1, x2, y2 = person["bbox"]
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 < 30 or y2 - y1 < 30:
                continue 
                
            crop = frame[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)
            
            # Use MTCNN to find a face within the person's bounding box
            face = self.mtcnn(pil_img)
            if face is not None:
                emb = self.resnet(face.unsqueeze(0).to(self.device)).detach().cpu()
                
                best_dist = float('inf')
                best_name = None
                for name, saved_emb in self.known_embeddings.items():
                    dist = (emb - saved_emb).norm().item()
                    if dist < best_dist:
                        best_dist = dist
                        best_name = name
                
                # Threshold for vggface2 is usually ~0.8 to 1.0 for a positive match
                if best_dist < 0.85:
                    matches.append(best_name)
                    
        return list(set(matches))
