import os
import csv
from transcriber import AudioTranscriber
from audio import Audio





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

                elif filename.endswith(".mp4"):
                    video = Video(full_file_path)
                    features = video.get_features()

                if features[0] is not None and features[0] != "-":
                    video_features.append([filename, folder] + features)

        # Escribir los datos en un archivo CSV
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File', 'Transcription', 'Classification', 'Sentiment', 'Emotion'])
            csv_writer.writerows(video_features)


# Ejemplo de uso:
if __name__ == "__main__":
    directory_path = 'preprocessed_dataset/'  # Reemplaza con la ruta de tu directorio
    output_csv = 'mp3_data.csv'  # Nombre del archivo CSV de salida

    feature_extractor = CsvGenerator(directory_path)
    feature_extractor.process_directory(output_csv)
