import numpy as np
from sklearn.cross_decomposition import CCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def check_data_structure(file_path):
    loaded_dict = np.load(file_path, allow_pickle=True).item()
    sample_keys = list(loaded_dict.keys())[:5]  # Adjust as needed to get a representative sample
    for key in sample_keys:
        print(f'{key}: {loaded_dict[key]}')


# Check the structure of the data in your files
check_data_structure('/Users/lernmi/Desktop/EmotionUnify/02_Model_Training/feature_extraction/text_features/dev_text_features.npy')
check_data_structure('/Users/lernmi/Desktop/EmotionUnify/02_Model_Training/feature_extraction/vision_features/train_video_features2.npy')


def load_and_filter_features(file_path, common_keys):
    loaded_dict = np.load(file_path, allow_pickle=True).item()
    filtered_dict = {k: v['features'] for k, v in loaded_dict.items() if k in common_keys}
    features_list = [v for k, v in filtered_dict.items()]
    if features_list:
        features_array = np.vstack(features_list)
        return features_array
    else:
        print(f'No features to stack for file {file_path}.')
        return None


# Load the original data
text_features_dict = np.load('/Users/lernmi/Desktop/EmotionUnify/02_Model_Training/feature_extraction/text_features/dev_text_features.npy', allow_pickle=True).item()
video_features_dict = np.load('/Users/lernmi/Desktop/EmotionUnify/02_Model_Training/feature_extraction/vision_features/train_video_features2.npy', allow_pickle=True).item()

# Find the common keys between the two datasets
common_keys = set(text_features_dict.keys()) & set(video_features_dict.keys())

# Load and filter the features using the common keys
text_features_train = load_and_filter_features('/Users/lernmi/Desktop/EmotionUnify/02_Model_Training/feature_extraction/text_features/dev_text_features.npy', common_keys)
video_features_train = load_and_filter_features('/Users/lernmi/Desktop/EmotionUnify/02_Model_Training/feature_extraction/vision_features/train_video_features2.npy', common_keys)

# Ensure both features arrays are not None before proceeding
if text_features_train is not None and video_features_train is not None:
    # Determine the minimum number of samples between the two modalities
    min_samples = min(text_features_train.shape[0], video_features_train.shape[0])

    # Truncate or select the corresponding number of samples from each modality
    text_features_train = text_features_train[:min_samples]
    video_features_train = video_features_train[:min_samples]

    # Correlation between modalities
    cca = CCA(n_components=2)  # You can choose the number of components
    cca.fit(text_features_train, video_features_train)

    text_c, video_c = cca.transform(text_features_train, video_features_train)

    # Correlation matrix from one dimension
    combined_features = np.hstack((text_features_train, video_features_train))  # Horizontally stack the feature arrays
    correlation_matrix = np.corrcoef(combined_features, rowvar=False)
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    # PCA Distribution
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_features)  # Use combined_features if you have audio features

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Result')
    plt.show()
else:
    print("One or both feature arrays are None, cannot proceed with analysis.")




