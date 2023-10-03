import torch
import configparser
import torchaudio
from transformers import Wav2Vec2Processor
from tqdm import tqdm


class AudioFeatureExtractor:
    def __init__(self, classification_type="Emotion"):
        config = configparser.ConfigParser()
        config.read('../config.ini')
        self.model_name = config.get('AudioFeatureExtraction', 'model_name')

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = EmotionModel.from_pretrained(self.model_name)
        self.classification_type = classification_type

    def mapping_emotion(self, df):
        # Your code for mapping emotions or other labels
        pass

    def extract_features_from_audio(self, audio_path, sampling_rate):
        waveform, _ = torchaudio.load(audio_path)
        return self.process_audio(waveform, sampling_rate)

    def process_audio(self, wavs, sampling_rate):
        y = self.processor([wav.cpu().numpy() for wav in wavs],
                           sampling_rate=sampling_rate,
                           return_tensors="pt",
                           padding="longest")

        y = y['input_values']

        with torch.no_grad():
            output = self.model(y)

        return {
            'hidden_states': output[0],
            'logits': output[1],
        }

    def extract_features_from_folder(self, audio_folder_path, df):
        audio_feature_dict = {}
        audio_files_to_process = [file for file in os.listdir(audio_folder_path) if file in set(df['audio_id'])]

        for audio_file in tqdm(audio_files_to_process):
            audio_path = os.path.join(audio_folder_path, audio_file)
            features = self.extract_features_from_audio(audio_path, 16000)  # Replace 16000 with actual sampling rate
            audio_feature_dict[audio_file] = features

        return audio_feature_dict

    def save_features(self, features, filename):
        torch.save(features, f'{filename}_audio_features.pth')

# Keep your existing EmotionModel and RegressionHead classes here


if __name__ == "__main__":
    feature_extractor = AudioFeatureExtractor("Emotion")
    audio_folder_path = "/path/to/audio/files"
    df = ...  # Your DataFrame

    extracted_features = feature_extractor.extract_features_from_folder(audio_folder_path, df)
    feature_extractor.save_features(extracted_features, "train")
