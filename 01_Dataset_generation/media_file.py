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
        return "custom"

    def get_file_type(self):
        return self.file_path[-3:]

    def get_file_name(self):
        return ["0", "0"]

    # TODO: HACER ADAPTADOR DE CADA CSV SEGUN SOURC
    def get_transcription_from_source(self, path):

        try:
            file_ids = self.get_file_name()

            df = pd.read_csv(path)
            #print(df)
            # Busca una fila donde ambas columnas 'Columna1' y 'Columna2' sean iguales a los valores de file_ids
            matching_row = df.loc[(df['Utterance_ID'] == file_ids[1]) & (df['Dialogue_ID'] == file_ids[0])]

            if not matching_row.empty:
                transcription = matching_row[
                    'Utterance']  # Reemplaza 'Utterance' con el nombre real de tu columna de transcripción
                return transcription
            return None
        except FileNotFoundError:
            return None

    def get_name_features(self):
        print("Entrando a la casita de get name features")
        file_name = os.path.basename(self.file_path)
        features = file_name[:-4].split("_")[:-1]
        print(f"FEATURES DE MIS HUEVOS: {features}")
        return features


    def get_transcription(self):
        if self.file_source != "custom":
            for csv_path in config.SOURCES_CSV_PATHS:
                transcription = self.get_transcription_from_source(csv_path)
        else:
            transcription = AudioTranscriber().transcribe(self.file_path)
            print("["*10,"Trans: ", transcription)
        return transcription

    def __str__(self):
        return f"Archivo: {self.file_path}, Tipo: {self.file_type}"

