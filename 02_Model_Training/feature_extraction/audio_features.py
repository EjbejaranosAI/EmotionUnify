import pandas as pd
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
import pickle
from tqdm import tqdm
from torchaudio.transforms import Resample

# Configuration
settings = {
    'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'resample_orig_freq': 48000,
    'resample_new_freq': 16000,
    'min_length': 16000
}

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class AudioFeatureExtractor:
    def __init__(self, settings, classification_type="Sentiment"):
        self.processor = Wav2Vec2Processor.from_pretrained(settings['model_name'])
        self.model = EmotionModel.from_pretrained(settings['model_name']).to(settings['device'])
        self.model.eval()
        self.resampler = Resample(settings['resample_orig_freq'], settings['resample_new_freq'])
        self.classification_type = classification_type


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
            try:  # Handling corrupted or problematic video/audio files
                video_path = os.path.join(video_folder_path, video_file)
                clip = VideoFileClip(video_path)
                audio = clip.audio.to_soundarray(fps=settings['resample_new_freq'])
                audio = audio.mean(axis=1)
                audio = self.resampler(torch.tensor(audio).float())

                input_values = self.processor(audio.cpu().numpy(), sampling_rate=settings['resample_new_freq'], return_tensors="pt").input_values
                input_values = input_values.to(settings['device'])

                with torch.no_grad():
                    hidden_states, logits = self.model(input_values)

                # Storing features as an array instead of a single value
                aggregated_features = hidden_states.squeeze().cpu().numpy()

                # Explicitly setting dtype for label as np.int
                label = np.array(df[df['video_id'] == video_file][f"{self.classification_type}_encoded"].values[0], dtype=np.int)

                audio_feature_dict[video_file.split('.')[0]] = {'features': aggregated_features, 'label': label}

            except Exception as e:
                print(f"Skipping {video_file} due to error: {e}")

        os.makedirs('./audio_features', exist_ok=True)
        with open(f'./audio_features/{set_name}_audio_features.pkl', 'wb') as f:
            pickle.dump(audio_feature_dict, f)



if __name__ == "__main__":
    feature_extractor = AudioFeatureExtractor(settings, classification_type="Sentiment")


    dataset_paths = {
        'Train': ('/01_Dataset_generation/dataset_adapters/MELD/train_splits', '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/train_sent_emo.csv'),
        'Test': ('/01_Dataset_generation/dataset_adapters/MELD/output_repeated_splits_test', '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/test_sent_emo.csv'),
        'Dev': ('/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete', '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_sent_emo.csv')
    }

    for set_name, (video_folder, csv_path) in dataset_paths.items():
        feature_extractor.extract_and_save_features(video_folder, csv_path, set_name)

