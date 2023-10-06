import numpy as np

def load_features(file_paths):
    features_dict = {}
    for modality, paths in file_paths.items():
        features_dict[modality] = {split: np.load(path, allow_pickle=True).item() for split, path in paths.items()}
    return features_dict


def prepare_data(features_dict, modalities, split):
    X_list, y_list = [], []
    for modality in modalities:
        features_labels = list(features_dict[modality][split].values())
        X = np.array([item['features'] for item in features_labels])
        y = np.array([item['label'] for item in features_labels])
        X_list.append(X)
        y_list.append(y)
    return np.concatenate(X_list, axis=1), np.concatenate(y_list, axis=0)


def merge_modalities(X_list):
    return np.concatenate(X_list, axis=1)