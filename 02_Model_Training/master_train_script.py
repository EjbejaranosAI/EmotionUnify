"""
THIS SCRIPT IS TO EXECUTE THE FEATURE EXTRACTION AND TRAINING OF THE MULTIMODAL MODELS,
IS GOING TO BE ORGANIZZED IN THE NEXT WAY THE PIPELINE:
0. Read configurations from config.ini
1. Read the metadata dataset and read the videos
2 Take video, audio or text modality for each video
3. From text modality made a feature extraction
    output: Save a NPY file with the feature extraction with the same size in tensor that the number of samples in csv
3. From audio modality made a feature extraction
     output: Save a NPY file with the feature extraction with the same size in tensor that the number of samples in csv
3. From video modality made a feature extraction
     output: Save a  NPY file with the feature extraction with the same size in tensor that the number of samples in csv
4. Take the feature extraction {Text, Vision, Audio} and concatenate the modalities (Two possible options[1.Just concatenate - 2. Network with attention mechanism])
5. Made prediction by two options (Classifier or BLSTM)
6. Store weigths and models
"""