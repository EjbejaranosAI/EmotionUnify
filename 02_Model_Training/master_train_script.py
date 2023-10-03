"""
THIS SCRIPT IS TO EXECUTE THE FEATURE EXTRACTION AND TRAINING OF THE MULTIMODAL MODELS,
IS GOING TO BE ORGANIZZED IN THE NEXT WAY THE PIPELINE:
0. Read configurations from config.ini
1. Read the metadata dataset and read the videos
2 Take video, audio or text modality for each video
3. From text modality made a feature extraction
    output: Save a NPY file with the feature extraction with the same size in tensor that the number of samples in csv
3. From audio modality made a feature extraction
     output: Save a NPY file with the feature extraction with the same size in tensor that the number of samples in csv
3. From video modality made a feature extraction
     output: Save a  NPY file with the feature extraction with the same size in tensor that the number of samples in csv
4. Take the feature extraction {Text, Vision, Audio} and concatenate the modalities (Two possible options[1.Just concatenate - 2. Network with attention mechanism])
5. Made prediction by two options (Classifier or BLSTM)
6. Store weigths and models
"""

import configparser
from feature_extraction import text_features, audio_features, video_features
from utilities import data_loader, read_metadata, store_models
from prediction import blstm_classifier, classifier

class MultimodalTrainPipeline:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def read_metadata_and_videos(self):
        """Read metadata and videos."""
        try:
            data_loader.load()
        except Exception as e:
            print(f"Error in read_metadata_and_videos: {e}")

    def extract_text_features(self):
        """Extract features from text modality."""
        try:
            text_features.extract()
        except Exception as e:
            print(f"Error in extract_text_features: {e}")

    def extract_audio_features(self):
        """Extract features from audio modality."""
        try:
            audio_features.extract()
        except Exception as e:
            print(f"Error in extract_audio_features: {e}")

    def extract_video_features(self):
        """Extract features from video modality."""
        try:
            video_features.extract()
        except Exception as e:
            print(f"Error in extract_video_features: {e}")

    def concatenate_features(self, mode='concatenate'):
        """Concatenate extracted features."""
        try:
            # Implement feature concatenation here
            pass
        except Exception as e:
            print(f"Error in concatenate_features: {e}")

    def make_prediction(self, mode='Classifier'):
        """Make predictions using Classifier or BLSTM."""
        try:
            if mode == 'Classifier':
                classifier.predict()
            else:
                blstm_classifier.predict()
        except Exception as e:
            print(f"Error in make_prediction: {e}")

    def store_weights_and_models(self):
        """Store model weights and configurations."""
        try:
            store_models.store()
        except Exception as e:
            print(f"Error in store_weights_and_models: {e}")

if __name__ == "__main__":
    pipeline = MultimodalTrainPipeline('config.ini')
    pipeline.read_metadata_and_videos()
    pipeline.extract_text_features()
    pipeline.extract_audio_features()
    pipeline.extract_video_features()
    pipeline.concatenate_features()
    pipeline.make_prediction()
    pipeline.store_weights_and_models()
