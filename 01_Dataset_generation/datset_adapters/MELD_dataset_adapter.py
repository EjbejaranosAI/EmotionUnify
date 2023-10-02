import os
import pandas as pd

class VideoProcessor:
    def __init__(self, csv_path, video_directory, output_directory):
        self.csv_path = csv_path
        self.video_directory = video_directory
        self.output_directory = output_directory

    def generate_new_filename(self, video_id, dialogue_id, utterance_id):
        return f"{video_id}_{dialogue_id}_{utterance_id}.mp4"

    def process_videos(self):
        # Carga el CSV en un DataFrame de pandas
        df = pd.read_csv(self.csv_path, delimiter='\t')  # Asegúrate de usar el delimitador correcto

        # Itera a través de las filas del DataFrame y procesa cada video
        for video_id, row in enumerate(df.itertuples(), start=1):
            dialogue_id = row.Dialogue_ID
            utterance_id = row.Utterance_ID

            # Genera el nuevo nombre de archivo
            new_filename = self.generate_new_filename(video_id, dialogue_id, utterance_id)

            # Ruta completa al archivo de origen
            source_video_path = os.path.join(self.video_directory, f'{dialogue_id}_{utterance_id}.mp4')

            # Ruta completa al directorio de destino
            destination_folder = os.path.join(self.output_directory, f'{dialogue_id}_{utterance_id}')

            # Ruta completa al archivo de destino
            destination_video_path = os.path.join(destination_folder, new_filename)

            try:
                # Verifica si el directorio de destino ya existe o crea uno nuevo
                os.makedirs(destination_folder, exist_ok=True)

                # Mueve el archivo a la carpeta de destino con el nuevo nombre
                os.rename(source_video_path, destination_video_path)
                print(f"Video {video_id}: Movido y renombrado como {new_filename}")
            except FileNotFoundError:
                print(f"Video {video_id}: Archivo no encontrado")
            except FileExistsError:
                print(f"Video {video_id}: El archivo de destino ya existe")

        print("Proceso completado.")

# Ejemplo de uso:
if __name__ == "__main__":
    csv_path = 'ruta/al/csv.csv'  # Reemplaza con la ruta de tu archivo CSV
    video_directory = 'directorio_de_videos'  # Reemplaza con la ruta donde se encuentran los archivos de video
    output_directory = 'videos_procesados'  # Reemplaza con la ruta de la carpeta de destino

    processor = VideoProcessor(csv_path, video_directory, output_directory)
    processor.process_videos()
