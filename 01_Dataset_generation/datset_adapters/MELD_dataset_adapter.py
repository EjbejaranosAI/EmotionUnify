import os
import pandas as pd
import os.path
from os import path
import logging
from tqdm import tqdm
import shutil
import subprocess
import tarfile
import shutil

#TODO: borrar cuando se importe el config y se recupere de ahi la ruta
OUTPUT_DIRECTORY = "../preprocessed_dataset"
# Function for printing a formatted title
def print_title(title):
    border = '═' * (len(title) + 2)
    print(f"\n    ╔{border}╗")
    print(f"    ║ {title} ║")
    print(f"    ╚{border}╝\n")


def print_global_summary(global_summary):
    print("\nGlobal Summary:")
    header = "║ Emotion/Sentiment  Train  Dev  Test  Total ║"
    header_len = len(header)

    # Calculate the width of the table dynamically
    table_width = header_len - 2  # Exclude the side borders

    print(f"╔{'═' * table_width}╗")
    print(f"║{'Emotion/Sentiment Summary'.center(table_width)}║")
    print(f"╠{'═' * table_width}╣")
    print(header)
    print(f"╠{'═' * table_width}╣")

    for emotion, counts in global_summary.items():
        row_str = f"║ {emotion: <16} {counts['train']: >5} {counts['dev']: >5} {counts['test']: >5} {counts['total']: >6} ║"
        print(row_str)

    print(f"╚{'═' * table_width}╝")


def download_meld_dataset():
    if not path.exists("./MELD"):
        os.mkdir("./MELD")

    try:
        logging.basicConfig(filename='download.log', level=logging.INFO)
        logging.info('Downloading MELD dataset')

        try:
            subprocess.run(['wget', 'https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz'])
        except:
            subprocess.run(['wget', 'http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'])

        # Use tqdm to track the progress of extracting files
        with tqdm(total=100, unit="%", desc="Extracting files", ascii=True) as progress_bar:
            subprocess.run(['tar', '-xvf', 'MELD.Raw.tar.gz'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            progress_bar.update(100)  # Mark as complete
            print("✅ Extraction complete.")

        os.remove('MELD.Raw.tar.gz')
        os.rename('MELD.Raw', 'MELD')

        files = os.listdir("./MELD")
        files_to_uncompress = []
        for file in files:
            if file.endswith('.tar.gz'):
                print("📦 tar.gz file found")
                files_to_uncompress.append(file)
        if len(files_to_uncompress) > 0:
            for file in files_to_uncompress:
                # Extract files and delete tar.gz files
                tar = tarfile.open("./MELD/"+file)
                with tqdm(total=100, unit="%", desc=f"Extracting {file}", ascii=True) as inner_progress_bar:
                    tar.extractall("./MELD/")
                    inner_progress_bar.update(100)  # Mark as complete
                tar.close()
                os.remove("./MELD/"+file)
                print(f"📦 {file} extraction complete.")
        else:
            print("No tar.gz files found")
    except Exception as e:
        logging.error('Error in downloading MELD dataset')
        print('❌ Error in downloading MELD dataset:', str(e))



class VideoProcessor:
    def __init__(self, csv_path, video_directory, output_directory, dataset_type, join_sets):
        self.csv_path = csv_path
        self.video_directory = video_directory
        self.join_sets = join_sets
        self.dataset_type = dataset_type if not join_sets else 'joined'
        self.output_directory = output_directory

        if os.path.exists(self.output_directory):
            shutil.rmtree(self.output_directory)
        os.makedirs(self.output_directory)

    def generate_new_filename(self, video_id, dialogue_id, utterance_id):
        return f"{video_id}_{dialogue_id}_{utterance_id}.mp4"

    def print_summary(self, classification_type):
        print(f"\n{classification_type} Summary:")
        summary_dict = {}
        for folder in os.listdir(self.output_directory):
            folder_path = os.path.join(self.output_directory, folder)
            if os.path.isdir(folder_path):
                num_files = len(os.listdir(folder_path))
                summary_dict[folder] = num_files

        max_key_length = max([len(key) for key in summary_dict.keys()])
        max_value_length = max([len(str(value)) for value in summary_dict.values()])
        total_length = max_key_length + max_value_length + 6  # 6 for extra spaces and special characters

        print(f"\n{'═' * (total_length + 2)}")
        print("║         Distribution           ║")
        print(f"{'═' * (total_length + 2)}")
        for key, value in summary_dict.items():
            print(f"║ {key: <{max_key_length}} {value: >{max_value_length}} ║")
        print(f"{'═' * (total_length + 2)}")

    def process_videos(self, classification_type, global_summary, join_datasets=False):
        df = pd.read_csv(self.csv_path)
        local_summary = {}

        for row in df.itertuples():
            dialogue_id = row.Dialogue_ID
            utterance_id = row.Utterance_ID
            video_id = f"dia{dialogue_id}_utt{utterance_id}.mp4"

            if classification_type == "Emotion":
                category = row.Emotion
            elif classification_type == "Sentiment":
                category = row.Sentiment
            else:
                print("❌ Invalid classification type.")
                return

            if join_datasets:
                destination_folder = os.path.join(self.output_directory, "joined", category)
            else:
                destination_folder = os.path.join(self.output_directory, self.dataset_type, category)

            os.makedirs(destination_folder, exist_ok=True)

            source_video_path = os.path.join(self.video_directory, video_id)
            destination_video_path = os.path.join(destination_folder, video_id)

            try:
                shutil.move(source_video_path, destination_video_path)
                local_summary[category] = local_summary.get(category, 0) + 1
                global_summary[category][self.dataset_type if not join_datasets else 'joined'] += 1
                global_summary[category]['total'] += 1
            except FileNotFoundError:
                print(f"❌ Video {video_id} not found.")

        print("Process completed.")


def adapt_meld_dataset():
    global_summary = {}

    csv_paths = ['./MELD/dev_sent_emo.csv', './MELD/train_sent_emo.csv', './MELD/test_sent_emo.csv']
    video_directories = ['./MELD/dev_splits_complete', './MELD/train_splits', './MELD/output_repeated_splits_test']
    dataset_types = ['dev', 'train', 'test']
    #TODO: cambiar el directorio cogiendolo del config
    output_directory =  OUTPUT_DIRECTORY

    print("Classification Types:")
    print("1. Emotion")
    print("2. Sentiment")
    choice = input("Choose classification type (1/2): ")
    classification_type = "Emotion" if choice == "1" else "Sentiment"  # Changed selection method

    for category in ["neutral", "joy", "sadness", "anger", "surprise", "disgust", "fear", "positive", "negative"]:
        global_summary[category] = {'train': 0, 'dev': 0, 'test': 0, 'total': 0}

    join_choice = input("Do you want to join train, dev, and test sets? (y/n): ")
    join_sets = True if join_choice.lower() == 'y' else False

    # Add 'joined' key to global_summary if datasets are to be joined
    if join_sets:
        for category in global_summary.keys():
            global_summary[category]['joined'] = 0

    for csv_path, video_directory, dataset_type in zip(csv_paths, video_directories, dataset_types):
        print(f"\nProcessing {csv_path} and videos from {video_directory}")
        processor = VideoProcessor(csv_path, video_directory, output_directory, dataset_type, join_sets)
        processor.process_videos(classification_type, global_summary)
    print_global_summary(global_summary)



def execute_option(option):
    if option == "1":
        print_title("Downloading MELD Dataset")
        download_meld_dataset()  # Uncomment this if you want to download the dataset
        print_title("Adapting MELD Dataset")
        adapt_meld_dataset()

    elif option == "2":
        print_title("Adapting MELD Dataset")
        adapt_meld_dataset()  # Call the function to adapt the dataset

    elif option == "3":
        print_title("Returning to Previous Option")

        # Delete the videos_procesados directory
        output_directory = OUTPUT_DIRECTORY
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        print(f"✅ Deleted directory: {output_directory}")

        # Delete the MELD directory
        meld_directory = './MELD'
        if os.path.exists(meld_directory):
            shutil.rmtree(meld_directory)
        print(f"✅ Deleted directory: {meld_directory}")

    else:
        print("❌ Invalid option.")



if __name__ == "__main__":
    print_title("🚀 Welcome to the PREPROCESSING PIPELINE! 🚀")
    print("📦 1️⃣ Download MELD dataset and adapt MELD 📥")
    print("🔄 2️⃣ Only Adapt MELD dataset to the framework 🛠️")
    print("🔙 3️⃣ Undo & Clean Up (Deletes processed directories)")

    option = input("👉 Select an option (1/2/3): 👈")
    execute_option(option)
