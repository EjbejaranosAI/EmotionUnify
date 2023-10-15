import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from tqdm import tqdm
import pandas as pd
import pickle
from codecarbon import EmissionsTracker
import mediapipe as mp



class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = F.softmax(self.fc(x), dim=0)
        print("Shape of weights:", weights.shape)
        weighted_sum = torch.sum(weights * x, dim=0)
        print("Shape of weighted_sum:", weighted_sum.shape)
        return weighted_sum

class VisionFeatureExtractor:
    def __init__(self, config,classification_type="Sentiment" ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = resnet18(pretrained=True).to(self.device)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))  # Removing the classification layer
        self.attention_layer = AttentionLayer(input_dim=512).to(self.device)  # Attention layer
        self.classification_type = classification_type
        self.config = config

    def mapping_labels(self, df):
        mapping = {
            'Emotion': {
                'neutral': [1, 0, 0, 0, 0, 0, 0],
                'surprise': [0, 1, 0, 0, 0, 0, 0],
                'fear': [0, 0, 1, 0, 0, 0, 0],
                'sadness': [0, 0, 0, 1, 0, 0, 0],
                'joy': [0, 0, 0, 0, 1, 0, 0],
                'disgust': [0, 0, 0, 0, 0, 1, 0],
                'anger': [0, 0, 0, 0, 0, 0, 1],
            },
            'Sentiment': {
                'neutral': [1, 0, 0],
                'positive': [0, 1, 0],
                'negative': [0, 0, 1]
            }
        }
        label_column = f"{self.classification_type}_encoded"
        df[label_column] = df[self.classification_type].map(mapping[self.classification_type])
        return df

    def extract_and_save_features(self, video_folder_path, video_path_csv, set_name):
        df = pd.read_csv(video_path_csv)
        df['video_id'] = "dia" + df['Dialogue_ID'].astype(str) + '_utt' + df['Utterance_ID'].astype(str) + '.mp4'
        df = self.mapping_labels(df)

        video_feature_dict = {}
        video_files_to_process = [file for file in os.listdir(video_folder_path) if file in set(df['video_id'])]

        for video_file in tqdm(video_files_to_process):
            video_path = os.path.join(video_folder_path, video_file)
            video_features = self.extract_features_from_video(video_path)

            if video_features is not None:
                label = np.squeeze(df[df['video_id'] == video_file][f"{self.classification_type}_encoded"].values)
                if label is None:
                    print(f"Skipping {video_file} due to None label.")
                    continue
                video_feature_dict[video_file.split('.')[0]] = {'features': video_features, 'label': label}
            torch.cuda.empty_cache()

        os.makedirs('./vision_features', exist_ok=True)
        with open(f'./vision_features/{set_name}_video_features_crop_face.pkl', 'wb') as f:
            pickle.dump(video_feature_dict, f)

    def extract_features_from_video(self, video_path):
        mp_face_detection = mp.solutions.face_detection

        cap = cv2.VideoCapture(video_path)
        frames = []
        all_features = []

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame is None:
                    print("Frame is None. Skipping this frame.")
                    continue

                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        cropped_face = frame[y:y+h, x:x+w]

                        if cropped_face.size == 0:
                            print("Cropped face is empty. Skipping this frame.")
                            continue

                        cropped_face = cv2.resize(cropped_face, (224, 224))
                        frames.append(cropped_face)
                        break

        if len(frames) == 0:
            print(f"No frames extracted from {video_path}. Skipping.")
            return None

        cap.release()

        for frame in frames:
            frame_tensor = torch.tensor(np.array(frame)).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                features = self.model(frame_tensor).squeeze(-1).squeeze(-1)  # [1, 512]
            all_features.append(features)

        # Convert to tensor and apply attention mechanism
        all_features_tensor = torch.stack(all_features).squeeze(1)  # Stack to make a [num_frames, 512] tensor
        print("Shape of all_features_tensor:", all_features_tensor.shape)

        aggregated_features = self.attention_layer(all_features_tensor).cpu().detach().numpy()  # Detach and convert to numpy

        return aggregated_features




if __name__ == "__main__":
    print("🚀 Initializing Vision Feature Extraction for Sentiment Classification... 🚀")
    times = []
    emissions_data = []

    feature_extractor = VisionFeatureExtractor(classification_type="Sentiment")

    # Example dataset paths; Replace with actual paths
    dataset_paths = {
        #'Train': ('/content/datasets/MELD/train_splits', '/content/datasets/MELD/train_sent_emo.csv'),
        'Test': ('/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/output_repeated_splits_test', '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/test_sent_emo.csv'),
        #'Dev': ('/content/datasets/MELD/dev_splits_complete', '/content/datasets/MELD/dev_sent_emo.csv')
    }

    for set_name, (video_folder, csv_path) in dataset_paths.items():
        tracker = EmissionsTracker(project_name=f"{set_name}_video_feature_extraction")
        start_time = time.time()
        tracker.start()

        feature_extractor.extract_and_save_features(video_folder, csv_path, set_name)

        current_time = time.time() - start_time
        total_emissions = tracker.stop()
        times.append(current_time)
        emissions_data.append(total_emissions)

        print(f"🔍 Total Emissions for {set_name}: {total_emissions} kg")
        print(f"⌛ Feature Extraction for {set_name} completed in {current_time:.2f} seconds ⌛")
        tracker.stop()