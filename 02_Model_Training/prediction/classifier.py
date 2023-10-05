import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_features(file_paths):
    features_dict = {}
    for modality, paths in file_paths.items():
        features_dict[modality] = {split: np.load(path, allow_pickle=True).item() for split, path in paths.items()}
    return features_dict


def prepare_data(features_dict, modalities):
    X_list, y_list = [], []
    for modality in modalities:
        for split in ['train', 'dev', 'test']:
            features_labels = list(features_dict[modality][split].values())
            X = np.array([item['features'] for item in features_labels])
            y = np.array([item['label'] for item in features_labels])
            X_list.append(X)
            y_list.append(y)
    return X_list, y_list


def merge_modalities(X_list):
    return np.concatenate(X_list, axis=1)


def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Classifier(X.shape[1], len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):  # Suponiendo 100 épocas
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            with torch.no_grad():
                outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')  # Ajusta el argumento 'average' según sea necesario
        print(f'Epoch {epoch + 1}, Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1 * 100:.2f}%')

    return model


def train_alternative_classifier(X, y):
    scaler = StandardScaler()  # Initialize the scaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = scaler.fit_transform(X_train)  # Scale the training data
    X_test = scaler.transform(X_test)
    # SVM
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    svm_preds = svm_classifier.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average='weighted')
    print(f'SVM - Accuracy: {svm_accuracy * 100:.2f}%, F1 Score: {svm_f1 * 100:.2f}%')

    # Random Forest
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    rf_preds = rf_classifier.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average='weighted')
    print(f'Random Forest - Accuracy: {rf_accuracy * 100:.2f}%, F1 Score: {rf_f1 * 100:.2f}%')

    linear_classifier = LogisticRegression(max_iter=10000)  # Increased max_iter for convergence
    linear_classifier.fit(X_train, y_train)
    linear_preds = linear_classifier.predict(X_test)
    linear_accuracy = accuracy_score(y_test, linear_preds)
    linear_f1 = f1_score(y_test, linear_preds, average='weighted')
    print(f'Linear Classifier - Accuracy: {linear_accuracy * 100:.2f}%, F1 Score: {linear_f1 * 100:.2f}%')


    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    xgb_preds = xgb_classifier.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds, average='weighted')
    print(f'XGBoost - Accuracy: {xgb_accuracy * 100:.2f}%, F1 Score: {xgb_f1 * 100:.2f}%')

    # For each classifier, compute and print precision and recall in addition to accuracy and F1 score
    for classifier, preds in zip(['SVM', 'Random Forest', 'Linear Classifier', 'XGBoost'],
                                 [svm_preds, rf_preds, linear_preds, xgb_preds]):
        precision = precision_score(y_test, preds, average='weighted')
        recall = recall_score(y_test, preds, average='weighted')
        print(f'{classifier} - Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%')


if __name__ == "__main__":
    file_paths = {
        # Define the file paths for each split and modality
        'text': {
            'train': '../feature_extraction/text_features/train_text_features.npy',
            'dev': '../feature_extraction/text_features/dev_text_features.npy',
            'test': '../feature_extraction/text_features/test_text_features.npy'
        }
        # ... (add other modalities if needed)
    }

    features_dict = load_features(file_paths)



    # For single modality classification
    for modality in file_paths.keys():
        print(f"Training classifier for {modality} modality...")
        X_list, y_list = prepare_data(features_dict, [modality])
        model = train_classifier(X_list[0], y_list[0])

    # Para clasificación alternativa
    for modality in file_paths.keys():
        print(f"Training alternative classifiers for {modality} modality...")
        X_list, y_list = prepare_data(features_dict, [modality])
        train_alternative_classifier(X_list[0], y_list[0])

    # For multi-modal classification by merging features
    #print("Training classifier for merged modalities...")
    #X_list, y_list = prepare_data(features_dict, file_paths.keys())
    #X_merged = merge_modalities(X_list)
    #model = train_classifier(X_merged, y_list[0])

    # ... (add other configurations if needed)
