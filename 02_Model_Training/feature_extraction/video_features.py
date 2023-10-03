import configparser
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16
import timm
from tqdm import tqdm
import pandas as pd


class VisionFeatureExtractor:
    def __init__(self, classification_type="Emotion"):
        config = configparser.ConfigParser()
        config.read('../config.ini')
        self.input_dim = config.getint('VisionFeatureExtraction', 'input_dim')
        self.output_dim = config.getint('VisionFeatureExtraction', 'output_dim')

        self.models = self.initialize_models()
        self.early_fusion_layer = EarlyFusionLayer(input_dim=self.input_dim, output_dim=self.output_dim)
        self.classification_type = classification_type

    def initialize_models(self):
        models = {
            'resnet': resnet18(pretrained=True),
            'vgg': vgg16(pretrained=True),
            'vit': timm.create_model("vit_base_patch16_224", pretrained=True)
        }
        for key in models:
            if key != 'vit':
                models[key] = nn.Sequential(*(list(models[key].children())[:-1]))
        return models

    def mapping_emotion(self, df):
        # Mapping emotions to labels
        pass

    def early_fusion(self, features_list):
        combined_features = np.concatenate(features_list, axis=1)
        combined_features_tensor = torch.tensor(combined_features).float()
        with torch.no_grad():
            fused_features = self.early_fusion_layer(combined_features_tensor)
        return fused_features.cpu().numpy()

    def extract_features_from_video(self, video_path):
        frames = self.extract_frames(video_path)
        features_list = []

        for model_name, model in self.models.items():
            frame_features = [self.extract_features_from_frame(model, frame) for frame in frames]
            features_list.append(np.stack(frame_features).reshape(len(frames), -1))

        return self.early_fusion(features_list)

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

    def extract_features_from_frame(self, model, frame):
        frame = cv2.resize(frame, (224, 224))
        tensor = torch.tensor(frame).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        with torch.no_grad():
            features = model(tensor)
        return np.squeeze(features.cpu().numpy())

    def extract_features_from_folder(self, video_folder_path, df):
        video_feature_dict = {}
        video_files_to_process = [file for file in os.listdir(video_folder_path) if file in set(df['video_id'])]

        for video_file in tqdm(video_files_to_process):
            video_path = os.path.join(video_folder_path, video_file)
            features = self.extract_features_from_video(video_path)
            video_feature_dict[video_file] = features

        return video_feature_dict

    def save_features(self, features, filename):
        np.save(f'{filename}_video_features.npy', features)


class EarlyFusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EarlyFusionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    feature_extractor = VisionFeatureExtractor("Emotion")
    video_folder_path = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/datset_adapters/MELD/dev_splits_complete"
    video_path_csv = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/datset_adapters/MELD/dev_sent_emo.csv""
    df = pd.read_csv(video_path_csv)

    extracted_features = feature_extractor.extract_features_from_folder(video_folder_path, df)
    feature_extractor.save_features(extracted_features, "train")
