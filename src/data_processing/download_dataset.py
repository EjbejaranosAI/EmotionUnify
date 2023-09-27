"""
This module is to download multimodal datasets and store them in the dataset directory
"""
import os
import logging
import shutil
import subprocess
import tarfile

def extract_all_tar_files(directory):
    """
    Recursively extract all .tar files within the given directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tar'):
                tar_path = os.path.join(root, file)
                with tarfile.open(tar_path, 'r') as tar:
                    tar.extractall(path=root)

def download_meld_dataset():
    """
    Download MELD dataset and store it into the dataset directory
    """
    os.chdir('./src/datasets')
    try:
        # Add logging configuration
        logging.basicConfig(filename='download.log', level=logging.INFO)
        logging.info('Downloading MELD dataset')

        # Use wget to download the dataset
        try:
            subprocess.run(['wget', 'https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz'])
        except:
            subprocess.run(['wget','http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz'])
        
        # Extract the downloaded dataset
        subprocess.run(['tar', '-xvf', 'MELD.Raw.tar.gz'])
        
        # Remove the downloaded tar.gz file
        os.remove('MELD.Raw.tar.gz')

        # Rename the extracted directory to "MELD"
        os.rename('MELD.Raw', 'MELD')

        # Extract the inner tar files
        files = os.listdir("./MELD")
        files_to_uncompress = []
        for file in files:
            if file.endswith('.tar.gz'):
                print("tar.gz file found")
                files_to_uncompress.append(file)
        if len(files_to_uncompress) >0:
            for file in files_to_uncompress:
                os.system("tar -xzf " + path + '/' + file + " -C " + path)
                os.remove(path + '/' + file)
        else:
            print("No tar.gz files found")





    except subprocess.CalledProcessError:
        logging.error("Failed to run command")
        print('Error in downloading MELD dataset')

if __name__ == "__main__":
    download_meld_dataset()
