import time
import configparser
from codecarbon import EmissionsTracker
from feature_extraction.text_features import TextFeatureExtractor
from feature_extraction.video_features import VisionFeatureExtractor
from feature_extraction.audio_features import AudioFeatureExtractor



class MultimodalTrainPipeline:
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.output_path = self.config['General'].get('output_path')
        classification_type = self.config['General'].get('classification_type', 'Sentiment')
        config_audio = self.config['AudioFeatureExtraction']
        self.text_feature_extractor = TextFeatureExtractor(classification_type)
        self.vision_feature_extractor = VisionFeatureExtractor(classification_type)
        self.audio_feature_extractor = AudioFeatureExtractor(config_audio,classification_type)


    def get_features(self, text_csv=None, vision_data=None, vision_csv=None,audio_data=None, audio_csv=None):
        text_features = None
        vision_features = None
        audio_features = None

        if text_csv is not None:
            times = []
            emissions_data = []
            tracker = EmissionsTracker(project_name=f"Text_feature_extraction")
            start_time = time.time()
            tracker.start()

            text_features = self.text_feature_extractor.extract_bert_features(text_csv)
            self.text_feature_extractor.save_features_pickle(text_features, self.output_path+'_text_features.pkl')

            current_time = time.time() - start_time
            total_emissions = tracker.stop()
            times.append(current_time)
            emissions_data.append(total_emissions)

        if audio_data is not None and audio_csv is not None:
            times = []
            emissions_data = []
            tracker = EmissionsTracker(project_name=f"Audio_feature_extraction")
            start_time = time.time()
            tracker.start()

            audio_features = self.audio_feature_extractor.extract_and_save_features(audio_data, audio_csv, "AudioData")

            current_time = time.time() - start_time
            total_emissions = tracker.stop()
            times.append(current_time)
            emissions_data.append(total_emissions)

        if vision_data is not None and vision_csv is not None:
            times = []
            emissions_data = []
            tracker = EmissionsTracker(project_name=f"Vision_feature_extraction")
            start_time = time.time()
            tracker.start()

            vision_features = self.vision_feature_extractor.extract_and_save_features(vision_data, vision_csv, "VisionData")

            current_time = time.time() - start_time
            total_emissions = tracker.stop()
            times.append(current_time)
            emissions_data.append(total_emissions)



    def align_features(self):
        pass
    def fusion_features(self):
        pass

    def made_prediction(self):
        pass

if __name__ == "__main__":
    pipeline = MultimodalTrainPipeline('./config.ini')
    dataset_text = './feature_extraction/text_features/dev_sent_emo.csv'
    dataset_vision = '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete'
    dataset_vision_csv = '/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_sent_emo.csv'
    dataset_audio = dataset_vision
    dataset_audio_csv = dataset_vision_csv

    print("Done [@ 1]")
    pipeline.get_features(
        text_csv=dataset_text,
        vision_data=dataset_vision,
        vision_csv=dataset_vision_csv,
        audio_data=dataset_audio,  # Assuming you have this
        audio_csv=dataset_audio_csv
    )
    print("Done [@ 2]")
    #pipeline.align_features()
    print("Done [@ 3]")
    #pipeline.fusion_features()
    print("Done [@ 4]")
    #pipeline.made_prediction()


