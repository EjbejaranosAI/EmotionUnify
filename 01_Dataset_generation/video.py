from transcriber import AudioTranscriber
from media_file import MediaFile
import config

class Video(MediaFile):
    def __init__(self, video_path):

        super().__init__(video_path)  # Llama al constructor de la clase base
        if not self.is_valid_video_format(video_path):
            raise ValueError(f"Error: El formato del archivo en '{video_path}' es incorrecto.")

    def is_valid_video_format(self, video_path):

        # Verificar si el formato del archivo es correcto
        valid_formats = ['mp4', 'avi']  # Agrega los formatos válidos según tus necesidades
        return video_path[-3:] in valid_formats


#TODO: HACER ADAPTADOR DE CADA CSV SEGUN SOURCE
    def get_transcription(self):
        if self.file_source != "custom":
            transcription = self.get_transcription_from_source()
        else:
            transcription = AudioTranscriber().transcribe(self.file_path)
        return transcription

    def get_features(self):
        features = None
        if self.file_source != "custom":
            features = [self.get_transcription()]

        return features


#v = Video("./dataset_adapters/MELD/output_repeated_splits_test/dia0_utt0.mp4")

#rint(v.get_transcription())
