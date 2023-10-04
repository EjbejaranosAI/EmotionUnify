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
        print("@@@@", self.file_source)
        if self.file_source != "custom":
            for csv_path in config.SOURCES_CSV_PATHS[self.file_source]:

                transcription = self.get_transcription_from_source(csv_path)
        else:
            transcription = AudioTranscriber().transcribe(self.file_path)
        print("[" * 10, "Trans: ", transcription)
        return transcription

    def get_features(self):
        features = None
        if self.file_source != "custom":
            features = [self.get_transcription()]
        print("["*10, "features: ", features)

        return features


v = Video("./datset_adapters/MELD/output_repeated_splits_test/._dia0_utt0.mp4")

print(v.get_transcription())
