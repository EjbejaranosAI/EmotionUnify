import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

from torchaudio.transforms import Resample
import torch.nn as nn
from tqdm import tqdm

# Configuration
config = {
    'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'resample_orig_freq': 48000,
    'resample_new_freq': 16000,
    'min_length': 16000
}

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
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
        self.model = EmotionModel.from_pretrained(config['model_name']).to(self.device)
        self.model.eval()
        self.resampler = Resample(int(config['resample_orig_freq']), int(config['resample_new_freq']))
        self.config = config

    def extract_features(self, audio_file_path):
        waveform, sample_rate = torchaudio.load(audio_file_path)
        waveform = self.resampler(waveform)

        input_values = self.processor(waveform.squeeze().numpy(), sampling_rate=config['resample_new_freq'], return_tensors="pt").input_values
        input_values = input_values.to(self.config['device'])

        with torch.no_grad():
            hidden_states, _ = self.model(input_values)

        aggregated_features = hidden_states.squeeze().cpu().numpy()
        return aggregated_features

if __name__ == "__main__":
    feature_extractor = AudioFeatureExtractor(config)
    audio_file_path = "/Users/lernmi/Desktop/EmotionUnify/03_Inference_Engine/feature_extractions/preprocessed_video_0.wav"
    extracted_features = feature_extractor.extract_features(audio_file_path)
    print(f"🎧 Extracted features shape: {extracted_features.shape} 🎧")