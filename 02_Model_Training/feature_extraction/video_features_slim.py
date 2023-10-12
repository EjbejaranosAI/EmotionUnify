import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16
import cv2
import timm
from tqdm import tqdm
import pandas as pd
import pickle
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt


class VisionFeatureExtractor:
    def __init__(self, output_dim=256, classification_type="Sentiment"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.initialize_model()
        self.output_dim = output_dim
        self.early_fusion_layer = EarlyFusionLayer(input_dim=self.model.feature_dim, output_dim=self.output_dim).to(
            self.device)
        self.classification_type = classification_type

    def initialize_model(self):
        model = resnet18(pretrained=True).to(self.device)
        model = nn.Sequential(*(list(model.children())[:-1]))  # Removing the classification layer
        model.feature_dim = 512  # Set the feature dimensionality of the model
        return model

    def initialize_models(self):
        models = {
            'resnet': resnet18(pretrained=True).to(self.device),
            'vgg': vgg16(pretrained=True).to(self.device),
            'vit': timm.create_model("vit_base_patch16_224", pretrained=True).to(self.device)
        }
        models['resnet'].feature_dim = 512
        models['vgg'].feature_dim = 4096  # Uncomment if using VGG
        models['vit'].feature_dim = 768  # Uncomment if using ViT
        for key in models:
            if key != 'vit':
                models[key] = nn.Sequential(*(list(models[key].children())[:-1]))
        return models

    def mapping_emotion(self, df):
        emotion_mapping = {
            'neutral': [1, 0, 0, 0, 0, 0, 0],
            'surprise': [0, 1, 0, 0, 0, 0, 0],
            'fear': [0, 0, 1, 0, 0, 0, 0],
            'sadness': [0, 0, 0, 1, 0, 0, 0],
            'joy': [0, 0, 0, 0, 1, 0, 0],
            'disgust': [0, 0, 0, 0, 0, 1, 0],
            'anger': [0, 0, 0, 0, 0, 0, 1]
        }
        df['Emotion_encoded'] = df['Emotion'].map(emotion_mapping)
        return df

    def mapping_sentiment(df):
        sentiment_mapping = {
            'neutral': [1, 0, 0],
            'positive': [0, 1, 0],
            'negative': [0, 0, 1]
        }
        df['Sentiment_encoded'] = df['Sentiment'].map(sentiment_mapping)
        return df

    def early_fusion(self, features_list):
        combined_features = np.concatenate(features_list, axis=1)
        combined_features_tensor = torch.tensor(combined_features).float().to(self.device)
        with torch.no_grad():
            fused_features = self.early_fusion_layer(combined_features_tensor)
        return fused_features.cpu().numpy()

    def extract_features_from_video(self, video_path):
        frames = self.extract_frames(video_path)
        frame_features = [self.extract_features_from_frame(frame) for frame in frames]
        features = np.stack(frame_features).reshape(len(frames), -1)
        return self.early_fusion([features])

    def extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def extract_features_from_frame(self, frame):
        frame = cv2.resize(frame, (224, 224))
        tensor = torch.tensor(frame).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        tensor = tensor.to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return np.squeeze(features.cpu().numpy())

    def extract_features_from_folder(self, video_folder_path, df):
        print("Entering into the extractor")
        video_feature_dict = {}
        video_files_to_process = [file for file in os.listdir(video_folder_path) if file in set(df['video_id'])]
        print(f"======{video_files_to_process}")
        if self.classification_type == "Emotion":
            df = self.mapping_emotion(df)
            label_column = 'Emotion_encoded'
        elif self.classification_type == "Sentiment":
            df = self.mapping_sentiment(df)
            label_column = 'Sentiment_encoded'
        for video_file in tqdm(video_files_to_process):
            video_path = os.path.join(video_folder_path, video_file)
            try:
                features = self.extract_features_from_video(video_path)
            except Exception as e:
                print(f"Failed to process video {video_file}: {e}")
                features = None
            label = df[df['video_id'] == video_file][label_column].values
            video_feature_dict[video_file] = {'video_id': video_file, 'label': label, 'features': features}
        return video_feature_dict

    def save_features(self, features, filename):
        os.makedirs('./vision_features', exist_ok=True)
        with open(f'./vision_features/{filename}_video_features.pkl', 'wb') as f:
            pickle.dump(features, f)


class EarlyFusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EarlyFusionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


def extract_and_save_features(feature_extractor, video_folder_path, video_path_csv, set_name):
    df = pd.read_csv(video_path_csv)
    df['video_id'] = "dia" + df['Dialogue_ID'].astype(str) + '_utt' + df['Utterance_ID'].astype(str) + '.mp4'
    print(f"Head of DataFrame for {set_name} set:")
    print(df.head(3))
    extracted_features = feature_extractor.extract_features_from_folder(video_folder_path, df)
    feature_extractor.save_features(extracted_features, set_name)


if __name__ == "__main__":
    print("🚀 Initializing Text Feature Extraction for Emotion Classification... 🚀")
    times = []
    emissions_data = []

    # Initialize Carbon tracker
    tracker = EmissionsTracker(project_name="video_feature_extraction")
    start_time = time.time()
    tracker.start()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = VisionFeatureExtractor(classification_type="Emotion")

    # Paths for Train set
    video_folder_path_train = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/train_splits"
    video_path_csv_train = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/train_sent_emo.csv"
    extract_and_save_features(feature_extractor, video_folder_path_train, video_path_csv_train, "train")
    current_time = time.time() - start_time
    emissions = tracker.stop()
    times.append(current_time)
    emissions_data.append(emissions)

    # Paths for Test set
    tracker.start()
    video_folder_path_test = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/output_repeated_splits_test"
    video_path_csv_test = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/test_sent_emo.csv"
    extract_and_save_features(feature_extractor, video_folder_path_test, video_path_csv_test, "test")
    current_time = time.time() - start_time
    emissions = tracker.stop()
    times.append(current_time)
    emissions_data.append(emissions)

    # Paths for Dev set
    tracker.start()
    video_folder_path_dev = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete"
    video_path_csv_dev = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_sent_emo.csv"
    extract_and_save_features(feature_extractor, video_folder_path_dev, video_path_csv_dev, "dev")
    current_time = time.time() - start_time
    emissions = tracker.stop()
    times.append(current_time)
    emissions_data.append(emissions)

    emissions: float = tracker.stop()
    print(f"🔍 Emissions: {emissions} kg")
    end_time = time.time()
    print(f"⌛ Feature Extraction completed in {end_time - start_time:.2f} seconds ⌛")


