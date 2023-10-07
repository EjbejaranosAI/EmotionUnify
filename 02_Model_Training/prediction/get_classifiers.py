import torch
from utils.utils_data import load_features, prepare_data, merge_modalities
from classifier_nn import train_classifier
from alternatives_classifiers import train_alternative_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    file_paths = {
        'text': {
            'train': '../feature_extraction/text_features/train_text_features.pkl',
            'dev': '../feature_extraction/text_features/dev_text_features.pkl',
            'test': '../feature_extraction/text_features/test_text_features.pkl'
        },
        'video': {
            'train': '../feature_extraction/vision_features/train_video_features.pkl',
            'dev': '../feature_extraction/vision_features/dev_video_features.pkl',
            'test': '../feature_extraction/vision_features/test_video_features.pkl'
        }
        # ... (add other modalities if needed)
    }

    features_dict = load_features(file_paths)

    # For single modality classification
    for modality in file_paths.keys():
        print(f"Training classifier for {modality} modality...")
        X_train, y_train = prepare_data(features_dict, [modality], 'train')
        X_dev, y_dev = prepare_data(features_dict, [modality], 'dev')
        X_test, y_test = prepare_data(features_dict, [modality], 'test')
        model = train_classifier(X_train, X_dev, X_test, y_train, y_dev, y_test)

    # Para clasificación alternativa
    for modality in file_paths.keys():
        print(f"Training alternative classifiers for {modality} modality...")
        X_train, y_train = prepare_data(features_dict, [modality], 'train')
        X_dev, y_dev = prepare_data(features_dict, [modality], 'dev')
        X_test, y_test = prepare_data(features_dict, [modality], 'test')
        train_alternative_classifier(X_train, X_dev, X_test, y_train, y_dev, y_test)
