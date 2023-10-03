import os
import csv
from transcriber import AudioTranscriber
from audio import Audio

class FeatureExtraction:
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

                if filename.endswith(".wav"):
                    # Parsear el nombre del archivo para extraer información
                    audio = Audio(full_file_path)

                    transcription = AudioTranscriber().transcribe(audio)
                    if transcription is not None and transcription != "-":
                        video_features.append([filename, transcription, folder] + audio.get_features())

        # Escribir los datos en un archivo CSV
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File', 'Transcription', 'Classification', 'Sentiment', 'Emotion'])
            csv_writer.writerows(video_features)


# Ejemplo de uso:
if __name__ == "__main__":
    directory_path = 'preprocessed_dataset/'  # Reemplaza con la ruta de tu directorio
    output_csv = 'mp3_data.csv'  # Nombre del archivo CSV de salida

    feature_extractor = FeatureExtraction(directory_path)
    feature_extractor.process_directory(output_csv)
