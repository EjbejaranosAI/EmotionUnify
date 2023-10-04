import os
import shutil
from tqdm import tqdm
import config
from video_preprocessing import VideoPreprocessing

class Preprocess:
    def __init__(self, path):
        self.path = path

    def create_folder_environment(self):
        source_path = self.path
        destination_path = os.path.join(os.getcwd(), "preprocessed_dataset")

        if os.path.exists(destination_path):
            files_in_destination = os.listdir(destination_path)
            if not config.SKIP_QUESTIONS and files_in_destination:
                response = input("║  The 'preprocessed_dataset' folder already exists and its NOT empty.\n"
                                 "║  ⚠️Do you want to remove the existing folder and regenerate it? (y/n): ")
                if "y" in response.lower():
                    shutil.rmtree(destination_path)
                else:
                    print("Skipped the creation of the folder environment.")

        folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]

        os.makedirs(destination_path, exist_ok=True)

        for folder in folders:
            destination_folder_path = os.path.join(destination_path, folder)
            os.makedirs(destination_folder_path, exist_ok=True)

    def preprocess_dataset(self):
        print("====> Down sampling data and moving it to the '/preprocessed_dataset' folder")
        source_path = self.path
        destination_path = os.path.join(os.getcwd(), "preprocessed_dataset")
        folders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
        with tqdm(total=len(folders), desc="Down sampling files", unit="folder") as pbar:
            for folder in folders:
                folder_path = os.path.join(source_path, folder)
                destination_folder_path = os.path.join(destination_path, folder)
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

                for file in files:
                    file_path = os.path.join(folder_path, file)
                    destination_file_path = os.path.join(destination_folder_path, file)
                    VideoPreprocessing(file_path).execute(destination_file_path)
                    break
                pbar.update(1)

    def execute(self):
        print("⚡ Let the preprocessing begin ⚡\n"
              "----------------------------------------------------\n")
        self.create_folder_environment()
        self.preprocess_dataset()
