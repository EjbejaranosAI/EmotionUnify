import os
import eel
from video_preprocessing import VideoPreprocessing
from audio import Audio
from transcriber import AudioTranscriber
from inference import make_prediction_from_samples


eel.init('web')
# Define the path of the folder you want to create
base_dir = "./web"
folder = "/preprocessed_output/"
transcription = None

# Check if the folder already exists
if not os.path.exists(base_dir+folder):
    # If it doesn't exist, create it
    os.makedirs(base_dir+folder)
    print(f"Folder created: {base_dir+folder}")
else:
    print(f"Folder already exists: {base_dir+folder}")

@eel.expose
def preprocess_video(file_path):
    print(os.getcwd())
    VideoPreprocessing(file_path).execute('./web/preprocessed_output/preprocessed_video.mp4')
    print(file_path)
    transcription = Audio("./web/preprocessed_output/preprocessed_video_0.wav").get_transcription()
    eel.set_preprocessing_media_output_paths("./preprocessed_output/preprocessed_video_0.mp4",
                                             "./preprocessed_output/preprocessed_video_0.wav", transcription)


@eel.expose
def inference(transcription):
    print("LLEGA AL INFERENCE")

    emotion = make_prediction_from_samples("./web/preprocessed_output/preprocessed_video_0.wav",
                                       "./web/preprocessed_output/preprocessed_video.mp4",
                                           transcription)
    eel.update_emotion(emotion)


eel.start('index2.html', size=(800, 1000))
