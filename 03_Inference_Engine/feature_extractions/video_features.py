import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = F.softmax(self.fc(x), dim=0)
        weighted_sum = torch.sum(weights * x, dim=0)
        return weighted_sum

class VisionFeatureExtractor:
    def __init__(self, config, classification_type="Sentiment"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet18(pretrained=True).to(self.device)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))  # Removing the classification layer
        self.attention_layer = AttentionLayer(input_dim=512).to(self.device)
        self.classification_type = classification_type
        self.config = config

    def extract_single_video_features(self, video_path):
        start_time = time.time()  # Start time

        cap = cv2.VideoCapture(video_path)
        all_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                print("Frame is None. Skipping this frame.")
                continue

            frame = cv2.resize(frame, (224, 224))
            frame_tensor = torch.tensor(np.array(frame)).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

            with torch.no_grad():
                features = self.model(frame_tensor).squeeze(-1).squeeze(-1)
            all_features.append(features)

        # Convert to tensor and apply attention mechanism
        all_features_tensor = torch.stack(all_features).squeeze(1)
        aggregated_features = self.attention_layer(all_features_tensor).cpu().detach().numpy()

        end_time = time.time()  # End time
        time_taken = end_time - start_time

        print(f"🕒 Time taken for feature extraction: {time_taken:.4f} seconds 🕒")
        print(f"📐 Dimension of extracted features: {aggregated_features.shape} 📐")

        return aggregated_features

# Example usage
if __name__ == "__main__":
    config = {}  # Your configuration here
    feature_extractor = VisionFeatureExtractor(config, classification_type="Sentiment")
    features = feature_extractor.extract_single_video_features("/Users/lernmi/Desktop/EmotionUnify/03_Inference_Engine/demo/video1.mp4")
