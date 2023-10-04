import os
import pandas as pd
from tabulate import tabulate
from csv_generator import CsvGenerator
import config
from preprocessing import Preprocess

# TODO: CHECK QUE LOS NOMBRES TENGAN TODOS EL MISMO NUMERO DE FEATURES Y QUE ESTAS SEAN VALIDAS
# TODO: PARA LAS FEATURES QUIZAS SE PODRÍA HACER UN CSV VINCULADO A CADA FILE
# TODO: GESTION DE ERRORES Y REINTENTOS EN PROCESAMIENTO DE ARCHIVOS
# TODO: FACILITAR INTERFAZ O ALGO PARA EVITAR PONER FPS, BITRATES... INCORRECTOS

def show_dataset_structure():
    
    dataset_path = 'dataset'
    print(80*"=")
    print(os.getcwd())
    print(80*"=")
    folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]
    table_data = []

    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        num_elements = len(os.listdir(folder_path))
        table_data.append([folder, num_elements])

    table = pd.DataFrame(table_data, columns=['Category', 'Number of samples'])
    table_str = tabulate(table, headers='keys', showindex=False, tablefmt='fancy_grid')
    print("         [DATASET STRUCTURE]")
    print(table_str)

    print("\n         [FILE NAME FEATURES]")
    try:
        folder_path = os.path.join(dataset_path, folders[0])
        file_name = os.listdir(folder_path)[0][:-4]
        print(f"Example of the filename features detected\n"
              f"    |Original file_name: {file_name}.mp4\n"
              f"    |Extracted features: {file_name.split('_')}")
    except:
        print(f"There are no files inside the {config.DATASET_FOLDER_PATH}")
    print("════════════════════════════════════════════════════ ")


def welcome_info():
    print("\n    ╔══════════════════════════════════════════════════╗          ")
    print("    ║      Welcome to the PREPROCESSING PIPELINE!      ║")
    print("    ╚══════════════════════════════════════════════════╝\n")
    print("║ This pipeline will preprocess your data for AI analysis 🤖 ║\n")

    print(f"    🚦Before starting, make sure there's a \"\{config.DATASET_FOLDER_PATH}\" folder in the \n"
          "    same \"main.py\" directory.")
    print(
        f"\n   Inside \"\config.DATASET_FOLDER_PATH\" folder, create as many folders as categories\n"
        "    you'd like to train your model with.\n"
        "    And inside those folders put the samples. You can add features in the name\n"
        "    separated by '_'\n")
    print("Example of dataset structure:")
    print(f"    /{config.DATASET_FOLDER_PATH}")
    print("        ├── /category1")
    print("        │   ├── file1_feature1_feature2_feature3.mp4")
    print("        │   ├── file2_feature1_feature2_feature3.mp4")
    print("        │   └── file3_feature1_feature2_feature3.mp4")
    print("        ├── /category2")
    print("        │   ├── file1_feature1_feature2_feature3.mp4")
    print("        │   ├── file2_feature1_feature2_feature3.mp4")
    print("        │   └── file3_feature1_feature2_feature3.mp4")
    print("        └── /category3")
    print("            ├── file1_feature1_feature2_feature3.mp4")
    print("            ├── file2_feature1_feature2_feature3.mp4")
    print("            └── file3_feature1_feature2_feature3.mp4")
    print("\n║===================     Let's get started! 🚀     ====================║\n")


def confirm_dataset():
    while True:
        response = input("Is the dataset loading correctly? (y/n): ")
        if 'y' in response.lower():
            return True
        elif 'n' in response.lower():
            print("Check you have correctly placed everything inside "
                  f"the {config.DATASET_FOLDER_PATH}\nfolder, check the file naming and features and try again...😢")
            return False
        else:
            print("Invalid response. Please enter 'y' or 'n'.\n")


def main():
    os.chdir("./01_Dataset_generation/")
    if not config.SKIP_EXPLANATION:
        welcome_info()

    show_dataset_structure()

    dataset_is_correct = True
    if not config.SKIP_QUESTIONS:
        dataset_is_correct = confirm_dataset()

    if dataset_is_correct:
        Preprocess(config.DATASET_FOLDER_PATH).execute()
        CsvGenerator(config.DESTINATION_FOLDER_PATH).execute()


if __name__ == '__main__':
    main()
