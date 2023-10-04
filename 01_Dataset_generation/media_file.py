import os
import config
import pandas as pd
from transcriber import AudioTranscriber

class MediaFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_type = self.get_file_type()
        self.file_source = self.determine_file_source()

#TODO: IMPLEMENTAR BIEN
    def determine_file_source(self):
        if "MELD" in self.file_path:
            return "meld"
        return "meld"

    def get_file_type(self):
        return self.file_path[-3:]

    def get_file_name(self):
        return ["0", "0"]

    # TODO: HACER ADAPTADOR DE CADA CSV SEGUN SOURC
    def get_transcription_from_source(self, path):
        print("BIEN DIREGIDA")
        try:
            file_ids = self.get_file_name()
            print("HASTA AQUI LLEGA", file_ids)
            df = pd.read_csv(path)
            #print(df)
            # Busca una fila donde ambas columnas 'Columna1' y 'Columna2' sean iguales a los valores de file_ids
            matching_row = df.loc[(df['Utterance_ID'] == file_ids[1]) & (df['Dialogue_ID'] == file_ids[0])]
            print(f"{matching_row}")
            print("HASTA AQUI LLEGA2")
            if not matching_row.empty:
                transcription = matching_row[
                    'Utterance']  # Reemplaza 'Utterance' con el nombre real de tu columna de transcripción
                return transcription
            return None
        except FileNotFoundError:
            return None

    def get_name_features(self):
        features = os.path.splitext(self.file_path)[0][-3:].split("_")
        return features


    def get_transcription(self):
        if self.file_source != "custom":
            for csv_path in config.SOURCES_CSV_PATHS:
                transcription = self.get_transcription_from_source(csv_path)
        else:
            transcription = AudioTranscriber().transcribe(self.file_path)
        return transcription

    def __str__(self):
        return f"Archivo: {self.file_path}, Tipo: {self.file_type}"

