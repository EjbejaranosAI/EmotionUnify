import os
from video_preprocessing import VideoPreprocessing
import eel

eel.init('web')

@eel.expose
def preprocess_video(file_path):
    print(os.getcwd())
    VideoPreprocessing(file_path).execute('./web/preprocessed_output/preprocessed_video.mp4')
    print(file_path)
    eel.set_preprocessing_media_output_paths("./preprocessed_output/preprocessed_video_0.mp4",
                                             "./preprocessed_output/preprocessed_video_0.wav")

@eel.expose
def extract_features(file_path):
    pass

eel.start('index2.html', size=(800, 1000))
