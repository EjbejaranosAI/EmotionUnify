import torch
from inference_model.baseline import FusionModels
from feature_extractions.audio_feature import AudioFeatureExtractor
from feature_extractions.video_features import VisionFeatureExtractor
from feature_extractions.text_feature import TextFeatureExtractor



model_path = "./inference_model/weigths"

def load_models_for_inference(path='./inference_model/weigths'):
    # Initialize models
    text_audio_model = FusionModels.TextAudioFusion()
    text_video_model = FusionModels.TextVideoFusion()
    final_model = FusionModels.FinalFusion()

    # Load saved state dictionaries
    text_audio_model.load_state_dict(torch.load(f'{path}/text_audio_model_best.pth'))
    text_video_model.load_state_dict(torch.load(f'{path}/text_video_model_best.pth'))
    final_model.load_state_dict(torch.load(f'{path}/final_model_best.pth'))

    # Put models in evaluation mode
    text_audio_model.eval()
    text_video_model.eval()
    final_model.eval()

    # Print models to verify
    print("🎵 Text-Audio Model Architecture 🎵")
    print(text_audio_model)
    print("\n🎥 Text-Video Model Architecture 🎥")
    print(text_video_model)
    print("\n🔗 Final Fusion Model Architecture 🔗")
    print(final_model)

    return text_audio_model, text_video_model, final_model





def make_inference(audio_sample, video_sample, text_sample, text_audio_model, text_video_model, final_model):
    index_to_sentiment = {
        0: 'Neutral',
        1: 'Positive',
        2: 'Negative'
    }
    with torch.no_grad():
        # Assuming the sample tensors are already on the same device as the models
        text_audio_fused = text_audio_model(audio_sample, text_sample)
        text_video_fused = text_video_model(video_sample, text_sample)
        final_output = final_model(text_audio_fused, text_video_fused)

        # Get the predicted class
        predicted_class = torch.argmax(final_output, dim=1)
        predicted_sentiment = index_to_sentiment[predicted_class.item()]

    return predicted_class, predicted_sentiment


def make_prediction_from_samples(audio_path, video_path, text_sentence, model_weights_path=model_path):
    print("OJITO",model_weights_path)
    # Configuration for audio feature extraction
    config = {
        'model_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resample_orig_freq': 48000,
        'resample_new_freq': 16000,
        'min_length': 16000
    }

    # Load fusion models
    text_audio_model, text_video_model, final_model = load_models_for_inference(model_weights_path)

    # Text feature extraction
    text_feature_extractor = TextFeatureExtractor(classification_type="Sentiment")
    text_sample_feature = text_feature_extractor.extract_single_sample_features(text_sentence)
    text_sample_feature = torch.tensor(text_sample_feature).float().unsqueeze(0).to(config['device'])

    # Audio feature extraction
    audio_feature_extractor = AudioFeatureExtractor(config)
    audio_sample_feature = audio_feature_extractor.extract_features(audio_path)
    audio_sample_feature = torch.tensor(audio_sample_feature).float().unsqueeze(0).to(config['device'])

    # Video feature extraction
    config_video = {}
    video_feature_extractor = VisionFeatureExtractor(config_video, classification_type="Sentiment")
    video_sample_feature = video_feature_extractor.extract_single_video_features(video_path)
    video_sample_feature = torch.tensor(video_sample_feature).float().unsqueeze(0).to(config['device'])

    # Make inference
    predicted_class, predicted_sentiment = make_inference(audio_sample_feature, video_sample_feature,
                                                          text_sample_feature, text_audio_model, text_video_model,
                                                          final_model)

    #return predicted_class.item(), predicted_sentiment
    return  predicted_sentiment



if __name__ == '__main__':
    text_sentence = "This is awesome you are the best men."
    audio_path = "/Users/lernmi/Desktop/EmotionUnify/03_Inference_Engine/feature_extractions/preprocessed_video_0.wav"
    video_path = "/Users/lernmi/Desktop/EmotionUnify/03_Inference_Engine/demo/video1.mp4"


    predicted_class, predicted_sentiment = make_prediction_from_samples(audio_path, video_path, text_sentence)
    print(f"\n🎯The predicted class is {predicted_class} which corresponds to a '{predicted_sentiment}' sentiment.🎯")