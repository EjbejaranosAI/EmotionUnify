import torch
from inference_model.baseline import FusionModels

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
    with torch.no_grad():
        # Assuming the sample tensors are already on the same device as the models
        text_audio_fused = text_audio_model(audio_sample, text_sample)
        text_video_fused = text_video_model(video_sample, text_sample)
        final_output = final_model(text_audio_fused, text_video_fused)

        # Get the predicted class
        predicted_class = torch.argmax(final_output, dim=1)

    return predicted_class



# Load models
text_audio_model, text_video_model, final_model = load_models_for_inference()


audio_sample_feature = torch.rand(1, 1024)  # Replace with the correct dimensions for your audio data
video_sample_feature = torch.rand(1, 512)   # Replace with the correct dimensions for your video data
text_sample_feature = torch.rand(1, 768)    # Replace with the correct dimensions for your text data



# Make inference
# Assuming audio_sample, video_sample, and text_sample are your preprocessed single samples
predicted_class = make_inference(audio_sample_feature, video_sample_feature, text_sample_feature, text_audio_model, text_video_model, final_model)



print("\n🎯 Predicted Class 🎯")
print(predicted_class)