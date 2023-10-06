import os
import sys

import config

def add_dataset():
    print("\n    === Add Dataset ===")
    print("      Available datasets:")
    i = 1
    for key in config.DATASETS_ADAPTERS_PATH.keys():
        print(f"        [{i}] {key}")

    dataset_choice = int(input("Select a dataset to add or press 'q' to abort: "))-1

    if dataset_choice == 'q':
        return  # Abort and return to the main menu

    selected_dataset = None
    selected_dataset_path = None
    counter = 0

    # Iterate through the key-value pairs in the dictionary
    for key, value in config.DATASETS_ADAPTERS_PATH.items():
        if counter == dataset_choice:
            selected_dataset = key
            selected_dataset_path = value
            break
        counter += 1

    if selected_dataset:
        print(f"Adding '{selected_dataset}' to the pipeline...")
        python_executable = "python3" if sys.version_info >= (3, 0) else "python"
        os.system(f"{python_executable} ./01_Dataset_generation{selected_dataset_path}")
    else:
        print("Invalid dataset choice. Please select a valid dataset.")

def run_preprocessing_pipeline():
    print("=== Run Preprocessing Pipeline ===")
    python_executable = "python3" if sys.version_info >= (3, 0) else "python"
    os.system(f"{python_executable} ./01_Dataset_generation/main.py")

def main():
    while True:
        print("\n    | 📊 01_Dataset_generation |")
        print("     -------------------------- ")
        print("        1. Add Dataset")
        print("        2. Run Preprocessing Pipeline")

        choice = input("Select an option (1/2) or press 'q' to quit: ")

        if choice == "1":
            add_dataset()
        elif choice == "2":
            run_preprocessing_pipeline()
        elif choice == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please select a valid option (1/2/3).")

if __name__ == "__main__":
    main()
