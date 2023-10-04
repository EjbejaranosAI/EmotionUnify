import os
import csv
from transcriber import AudioTranscriber
from audio import Audio
from video import Video

class CsvGenerator:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def execute(self, output_csv="videos_features.csv"):
        video_features = []

        # Iterar sobre los archivos y directorios en el directorio principal
        for folder in os.listdir(self.directory_path):
            if folder == ".DS_Store":
                continue
            for filename in os.listdir(os.path.join(self.directory_path, folder)):
                full_file_path = os.path.join(self.directory_path, folder, filename)
                features = []
                if filename.endswith(".wav"):
                    # Parsear el nombre del archivo para extraer información
                    audio = Audio(full_file_path)
                    features = audio.get_features()
                    #print(80*"*")
                    #print("----->", features)

                elif filename.endswith(".mp4"):
                    video = Video(full_file_path)
                    features = video.get_features()

                if features is not None and len(features) >= 1 and features[0] is not None:
                    video_features.append([filename[:-4], folder] + features)

        # Escribir los datos en un archivo CSV
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File', 'Classification', 'Transcription'])
            csv_writer.writerows(video_features)
