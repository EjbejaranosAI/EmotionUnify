from torchaudio.transforms import Resample
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel
import os
import pandas as pd
import numpy as np


# Custom Regression Head
class RegressionHead(nn.Module):
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


# Custom Model that includes Wav2Vec2 and Regression Head
class CustomEmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# Configuration
settings = {
    'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
    'device': 'cpu',
    'resample_orig_freq': 48000,
    'resample_new_freq': 16000,
    'min_length': 16000
}

# Initialize model and processor
processor = Wav2Vec2Processor.from_pretrained(settings['model_name'])
model = CustomEmotionModel.from_pretrained(settings['model_name']).to(settings['device'])

# Initialize resampler
resampler = Resample(settings['resample_orig_freq'], settings['resample_new_freq'])


# Helper functions
def read_and_preprocess_audio(file_path):
    audio_data, _ = sf.read(file_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data_resampled = resampler(torch.tensor(audio_data).float())
    if audio_data_resampled.shape[0] < settings['min_length']:
        audio_data_resampled = nnf.pad(audio_data_resampled,
                                       (0, settings['min_length'] - audio_data_resampled.shape[0]))
    return audio_data_resampled


def process_audio(audio_data):
    audio_input = processor(audio_data.cpu().numpy(), sampling_rate=settings['resample_new_freq'], return_tensors="pt",
                            padding="longest")['input_values']
    audio_input = audio_input.to(settings['device'])
    with torch.no_grad():
        outputs = model(audio_input)
    return {'hidden_states': outputs[0], 'logits': outputs[1]}


def extract_features_and_labels(audio_dir, csv_path, classification_type):
    feature_dict = {}
    df = pd.read_csv(csv_path)
    df['video_id'] = "dia" + df['Dialogue_ID'].astype(str) + '_utt' + df['Utterance_ID'].astype(str) + '.wav'

    if classification_type == "Emotion":
        df = mapping_emotion(df)
    elif classification_type == "Sentiment":
        df = mapping_sentiment(df)

    label_column = f"{classification_type}_encoded"

    for audio_file in tqdm(os.listdir(audio_dir)):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(audio_dir, audio_file)
            audio_data = read_and_preprocess_audio(audio_path)
            features = process_audio(audio_data)
            label = df[df['video_id'] == audio_file][label_column].values[0]
            feature_dict[audio_file] = {'features': features['hidden_states'].detach().cpu().numpy(), 'label': label}
    return feature_dict


if __name__ == "__main__":
    csv_train = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/train_sent_emo.csv"
    csv_dev = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_sent_emo.csv"
    csv_test = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/test_sent_emo.csv"

    feature_dict_train = extract_features_and_labels(
        "/01_Dataset_generation/dataset_adapters/MELD/train_splits_complete_wav", csv_train,
                                                     "Emotion")
    np.save('./audio_features/audio__train_features.npy', feature_dict_train)

    feature_dict_dev = extract_features_and_labels(
        "/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete_wav", csv_dev,
                                                   "Emotion")
    np.save('./audio_features/audio__dev_features.npy', feature_dict_dev)

    feature_dict_test = extract_features_and_labels(
        "/01_Dataset_generation/dataset_adapters/MELD/output_repeated_splits_test_wav", csv_test,
                                                    "Emotion")
    np.save('./audio_features/audio__test_features.npy', feature_dict_test)

