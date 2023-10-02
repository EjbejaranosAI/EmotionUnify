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
    
    if option == "1":
        print(f"Ejecutando 01_Dataset_generation con {python_executable}...")
        os.system(f"{python_executable} ./01_Dataset_generation/main.py")
    elif option == "2":
        print(f"Ejecutando 02_Model_Training con {python_executable}...")
        os.system(f"{python_executable} ./02_Model_Training/main.py")
    elif option == "3":
        print(f"Ejecutando 03_Inference_Engine con {python_executable}...")
        os.system(f"{python_executable} ./03_Inference_Engine/main.py")
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")

def main():
    install_dependencies()
    print("===================== Menú de Opciones =====================")
    print("1. Ejecutar 01_Dataset_generation")
    print("2. Ejecutar 02_Model_Training")
    print("3. Ejecutar 03_Inference_Engine")

    option = input("Seleccione una opción (1/2/3): ")
    execute_option(option)

if __name__ == "__main__":
    main()
