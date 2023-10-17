import time
import configparser
from codecarbon import EmissionsTracker
from feature_extraction.text_features import TextFeatureExtractor
from feature_extraction.video_features import VisionFeatureExtractor
from feature_extraction.audio_features import AudioFeatureExtractor


class MultimodalTrainPipeline:
    def __init__(self, config_file=None):
        self.load_config(config_file)

    def load_config(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.output_path = self.config['General'].get('output_path')
        classification_type = self.config['General'].get('classification_type', 'Sentiment')
        config_audio = self.config['AudioFeatureExtraction']
        self.text_feature_extractor = TextFeatureExtractor(classification_type)
        self.vision_feature_extractor = VisionFeatureExtractor(classification_type)
        self.audio_feature_extractor = AudioFeatureExtractor(config_audio, classification_type)

    def track_emissions_and_time(self, func, *args, **kwargs):
        tracker = EmissionsTracker(project_name=f"{func.__name__}")
        start_time = time.time()
        tracker.start()

        result = func(*args, **kwargs)

        elapsed_time = time.time() - start_time
        total_emissions = tracker.stop()
        print(80*'==')
        print(f"⏱ Time elapsed for {func.__name__}: {elapsed_time} seconds")
        print(f"🌍 Total emissions: {total_emissions} kg CO2")

        return result

    def get_features(self, text_csv=None, vision_data=None, vision_csv=None, audio_data=None, audio_csv=None):
        if text_csv:
            features = self.track_emissions_and_time(self.text_feature_extractor.extract_bert_features, text_csv)
            self.text_feature_extractor.save_features_pickle(features, f"{self.output_path}_text_features.pkl")

        if audio_data and audio_csv:
            self.track_emissions_and_time(self.audio_feature_extractor.extract_and_save_features, audio_data, audio_csv,
                                          "AudioData")

        if vision_data and vision_csv:
            self.track_emissions_and_time(self.vision_feature_extractor.extract_and_save_features, vision_data,
                                          vision_csv, "VisionData")

    def align_features(self):
        pass
    def fusion_features(self):
        pass

    def made_prediction(self):
        pass

if __name__ == "__main__":
    print("🚀 Initializing... [@ 1]")
    pipeline = MultimodalTrainPipeline('./config.ini')
    dataset_text = './feature_extraction/text_features/dev_sent_emo.csv'
    dataset_vision = '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete'
    dataset_vision_csv = '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_sent_emo.csv'
    dataset_audio = dataset_vision
    dataset_audio_csv = dataset_vision_csv

    pipeline.get_features(
        text_csv=dataset_text,
        vision_data=dataset_vision,
        vision_csv=dataset_vision_csv,
        audio_data=dataset_audio,
        audio_csv=dataset_audio_csv
    )
    print("🎉 Feature extraction complete! [@ 2]")
    #pipeline.align_features()
    print("Done [@ 3]")
    #pipeline.fusion_features()
    print("Done [@ 4]")
    #pipeline.made_prediction()


