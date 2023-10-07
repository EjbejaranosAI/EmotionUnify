import pandas as pd
from pydub import AudioSegment
import os
from tqdm import tqdm
def convert_videos_to_wav(input_folder, output_folder, csv_file_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(csv_file_path)
    df['video_id'] = 'dia' + df['Dialogue_ID'].astype(str) + '_utt' + df['Utterance_ID'].astype(str)

    failed_conversions = []  # To keep track of failed conversions

    for video_id in tqdm(df['video_id']):
        mp4_filename = f"{video_id}.mp4"
        mp4_path = os.path.join(input_folder, mp4_filename)
        wav_path = os.path.join(output_folder, mp4_filename.replace(".mp4", ".wav"))

        try:
            if os.path.exists(mp4_path):  # Check if the video file exists
                audio = AudioSegment.from_file(mp4_path, format="mp4")
                audio.export(wav_path, format="wav")
                print(f"Converted {mp4_filename} to {wav_path}.")
            else:
                print(f"{mp4_path} does not exist. Skipping.")
                failed_conversions.append(mp4_filename)
        except Exception as e:
            print(f"Could not convert {mp4_filename} due to {e}. Skipping.")
            failed_conversions.append(mp4_filename)

    print(f"Failed conversions: {failed_conversions}")



# Train
video_directory_train = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/train_splits"
csv_file_train = "/01_Dataset_generation/dataset_adapters/MELD/train_sent_emo.csv"
output_directory_train = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/train_splits_complete_wav"
convert_videos_to_wav(video_directory_train, output_directory_train, csv_file_train)

# Dev
video_directory_dev = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete"
csv_file_dev = "/01_Dataset_generation/dataset_adapters/MELD/dev_sent_emo.csv"
output_directory_dev = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/dev_splits_complete_wav"
convert_videos_to_wav(video_directory_dev, output_directory_dev, csv_file_dev)

# Test
video_directory_test = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/output_repeated_splits_test"
csv_file_test = "/01_Dataset_generation/dataset_adapters/MELD/test_sent_emo.csv"
output_directory_test = "/Users/lernmi/Desktop/EmotionUnify/01_Dataset_generation/dataset_adapters/MELD/output_repeated_splits_test_wav"
convert_videos_to_wav(video_directory_test, output_directory_test, csv_file_test)