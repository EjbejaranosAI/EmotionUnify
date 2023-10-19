import os
import re

import config
import pandas as pd
from transcriber import AudioTranscriber

class MediaFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_type = self.get_file_type()
        self.file_source = self.determine_file_source()

    def is_meld_file(self, filename):
        pattern = r".*dia\d+_utt\d+.*"
        return re.match(pattern, filename) is not None

    def determine_file_source(self):
        file_name = os.path.basename(self.file_path)
        if self.is_meld_file(file_name):
            return "meld"
        return "custom"

    def get_file_type(self):
        return self.file_path[-3:]

    def get_file_ids(self):
        file_ids = None
        if self.file_source == "meld":
            file_name = os.path.basename(self.file_path)
            file_ids_str = file_name[:-4].replace("dia","").replace("utt","").split("_")
            # Convertir elementos de la lista en enteros
            file_ids = [int(id_str) for id_str in file_ids_str]
        else:
            print(f"Aun no se ha implementado la extracción de file ids para {self.file_source}")
        return file_ids


    # TODO: HACER ADAPTADOR DE CADA CSV SEGUN SOURC
    def get_transcription_from_source(self):
        try:
            file_ids = self.get_file_ids()
            for csv_path in config.SOURCES_CSV_PATHS[self.file_source]:
                try:
                    df = pd.read_csv(csv_path)

                    # Busca una fila donde ambas columnas 'Columna1' y 'Columna2' sean iguales a los valores de file_ids
                    matching_row = df.loc[(df['Utterance_ID'] == file_ids[1]) & (df['Dialogue_ID'] == file_ids[0])]

                    if not matching_row.empty:
                        transcription = matching_row['Utterance']  # Reemplaza 'Utterance' con el nombre real de tu columna de transcripción
                        transcription = str(transcription.iloc[0])
                        return transcription
                    else:
                        print("Not matching rows found in:", csv_path)
                except FileNotFoundError:
                    print(f"File not found: {csv_path}")
                except Exception as e:
                    print(f"Error processing CSV: {csv_path}, Error: {str(e)}")

            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def get_name_features(self):
        file_name = os.path.basename(self.file_path)
        features = file_name[:-4].split("_")[:-1]
        return features


    def get_transcription(self):
        if self.file_source != "custom":
            transcription = self.get_transcription_from_source()
        else:
            transcription = AudioTranscriber().transcribe(self.file_path)
        return transcription

    def __str__(self):
        return f"Archivo: {self.file_path}, Tipo: {self.file_type}"


