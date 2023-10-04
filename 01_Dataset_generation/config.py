SKIP_EXPLANATION = False
SKIP_QUESTIONS = False  # SKIP AT YOUR OWN RISK

DATASET_FOLDER_PATH = "dataset"
DESTINATION_FOLDER_PATH = "preprocessed_dataset"

# VIDEO
TARGET_VIDEO_FPS = 3
TARGET_VIDEO_RESOLUTION_SCALE = 0.7
SPLIT_VIDEO_WHEN_THERES_SILENCE = True
MIN_VIDEO_CHUNK_TIME = 0.5
# AUDIO
REMOVE_AUDIO_FROM_PREPROCESSED_VIDEOS = False
TARGET_AUDIO_SAMPLERATE = 8000
TARGET_AUDIO_BITRATE = "128k"

# DATASETS
SOURCES_CSV_PATHS = {
    "meld": [
        "./datset_adapters/MELD/train_sent_emo.csv"
    ]
}
# good :
# -trans: me fue muy bien (positive)
# -img: :) (positive)

# video_splits :
# -trans: me fue muy bien (negative)
# -img: :( (negative)
