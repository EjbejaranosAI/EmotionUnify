import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel
import os
import pickle
from tqdm import tqdm
from torchaudio.transforms import Resample
import torch
import torch.nn as nn

# Configuration
config = {
    'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'resample_orig_freq': 48000,
    'resample_new_freq': 16000,
    'min_length': 16000
}
#TODO:REFACTOR
class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

class AudioFeatureExtractor:
    def __init__(self, config, classification_type="Sentiment"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
        self.model = EmotionModel.from_pretrained(config['model_name']).to(self.device)
        self.model.eval()
        self.resampler = Resample(int(config['resample_orig_freq']), int(config['resample_new_freq']))
        self.classification_type = classification_type
        self.carbon_emissions = 0.0  # Initialize carbon emissions counter
        self.power_rating = 200.0  # in watts
        self.conversion_factor = 0.0002  # in kg CO2 per joule
        self.emission_data = []  # Initialize list to store emissions data
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

        audio_feature_dict = {}
        video_files_to_process = [file for file in os.listdir(video_folder_path) if file in set(df['video_id'])]

        for video_file in tqdm(video_files_to_process):
            start_time = time.time()  # Start time

            try:
                video_path = os.path.join(video_folder_path, video_file)
                clip = VideoFileClip(video_path)
                audio = clip.audio.to_soundarray(fps=config['resample_new_freq'])
                audio = audio.mean(axis=1)
                audio = self.resampler(torch.tensor(audio).float())

                input_values = self.processor(audio.cpu().numpy(), sampling_rate=config['resample_new_freq'], return_tensors="pt").input_values
                input_values = input_values.to(config['device'])

                with torch.no_grad():
                    hidden_states, logits = self.model(input_values)

                aggregated_features = hidden_states.squeeze().cpu().numpy()
                label = np.array(df[df['video_id'] == video_file][f"{self.classification_type}_encoded"].values[0], dtype=np.int)

                audio_feature_dict[video_file.split('.')[0]] = {'features': aggregated_features, 'label': label}

            except Exception as e:
                print(f"Skipping {video_file} due to error: {e}")

            end_time = time.time()  # End time
            time_taken = end_time - start_time

            energy_consumed = self.power_rating * time_taken  # Energy in joules
            carbon_emitted = energy_consumed * self.conversion_factor  # Carbon emissions in kg CO2

            self.carbon_emissions += carbon_emitted
            self.emission_data.append(self.carbon_emissions)  # Store emissions data

        os.makedirs('./audio_features', exist_ok=True)
        with open(f'./feature_extraction/audio_features/{set_name}_audio_features.pkl', 'wb') as f:
            pickle.dump(audio_feature_dict, f)

        # Plotting carbon emissions
        plt.plot(self.emission_data)
        plt.xlabel('Number of Videos Processed')
        plt.ylabel('Total Carbon Emissions (kg CO2)')
        plt.title('Carbon Emissions for Audio Feature Extraction')
        plt.show()

if __name__ == "__main__":
    feature_extractor = AudioFeatureExtractor(config, classification_type="Sentiment")
    dataset_paths = {
        #'Train': ('/01_Dataset_generation/dataset_adapters/MELD/train_splits', '/01_Dataset_generation/dataset_adapters/MELD/train_sent_emo.csv'),
        'Test': ('/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/output_repeated_splits_test', '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/test_sent_emo.csv'),
        #'Dev': ('/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete', '/01_Dataset_generation/dataset_adapters/MELD/dev_sent_emo.csv')
    }

    for set_name, (video_folder, csv_path) in dataset_paths.items():
        feature_extractor.extract_and_save_features(video_folder, csv_path, set_name)
