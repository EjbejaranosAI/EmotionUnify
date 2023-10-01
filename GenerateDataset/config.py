SKIP_EXPLANATION = True
SKIP_QUESTIONS = True  # SKIP AT YOUR OWN RISK

DATASET_FOLDER_PATH = "dataset"
DESTINATION_FOLDER_PATH = "preprocessing_files"

#VIDEO
TARGET_VIDEO_FPS = 1
TARGET_VIDEO_RESOLUTION_SCALE = 0.2
SPLIT_VIDEO_WHEN_THERES_SILENCE = True
MIN_VIDEO_CHUNK_TIME = 0.5
#AUDIO
TARGET_AUDIO_SAMPLERATE = 8000
TARGET_AUDIO_BITRATE = "128k"



#good :
# -trans: me fue muy bien (positive)
# -img: :) (positive)

#bad :
# -trans: me fue muy bien (negative)
# -img: :( (negative)
