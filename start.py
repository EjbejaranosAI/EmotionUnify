import os
import sys


def install_dependencies():
    print("==== Instalando dependencias ====")
    platform = sys.platform.lower()
    if "darwin" in platform:  # macOS
        os.system("brew install ffmpeg")
    elif "linux" in platform:  # Linux
        os.system("sudo apt-get install ffmpeg")
    elif "win" in platform:  # Windows
        os.system("choco install ffmpeg")
    else:
        print("Sistema operativo no compatible. Por favor, instale FFmpeg manualmente.")
        return

    os.system("pip install -q -r ./01_Dataset_generation/requirements.txt")


def execute_option(option):
    python_executable = "python3" if sys.version_info >= (3, 0) else "python"
    success = True
    if option == "1":
        print(f"Executing 01_Dataset_generation with {python_executable}... 🚀")
        os.system(f"{python_executable} ./01_Dataset_generation/start_dataset_generation.py")
    elif option == "2":
        print(f"Executing 02_Model_Training with {python_executable}... 🏋️")
        os.system(f"{python_executable} ./02_Model_Training/main.py")
    elif option == "3":
        print(f"Executing 03_Inference_Engine with {python_executable}... 🔮")
        os.system(f"{python_executable} ./03_Inference_Engine/main.py")
    else:
        success = False
        print("Invalid option. Please select a valid option. ❌")

    return success


def main():
    while True:
        print("===================== Options Menu =====================")
        print("📊 1. Execute 01_Dataset_generation")
        print("🧠 2. Execute 02_Model_Training")
        print("🔍 3. Execute 03_Inference_Engine")

        option = input("Select an option (1/2/3) or press 'q' to quit: ")

        if option.lower() == 'q':
            break

        if execute_option(option):
          break


if __name__ == "__main__":
    main()
