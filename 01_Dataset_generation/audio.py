import os
from transcriber import AudioTranscriber
import pandas as pd
from media_file import MediaFile
import config

class Audio(MediaFile):
    def __init__(self, audio_path):
        super().__init__(audio_path)  # Llama al constructor de la clase base
        if not self.is_valid_audio_format(audio_path):
            raise ValueError(f"Error: El formato del archivo en '{audio_path}' es incorrecto.")

    def is_valid_audio_format(self, audio_path):
        # Verificar si el formato del archivo es correcto
        valid_formats = ['mp3', 'wav', 'ogg']  # Agrega los formatos válidos según tus necesidades
        return self.file_type in valid_formats

    def get_features(self):
        features = []
        features.append(self.get_transcription())
        if self.file_source == "custom":
            features.append(self.get_name_features())
        return features

a = Audio("dia49_utt4.mp4")

print(a.get_transcription())
