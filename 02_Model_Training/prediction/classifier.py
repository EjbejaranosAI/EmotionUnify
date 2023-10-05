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


def train_classifier(X_train, X_dev, X_test, y_train, y_dev, y_test):
    # Prepare DataLoader for training, development, and test sets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
    dev_dataset = TensorDataset(torch.tensor(X_dev, dtype=torch.float), torch.tensor(y_dev, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Classifier(X_train.shape[1], len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):  # Assuming 100 epochs
        # Training code
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluation on development set
        model.eval()
        all_preds, all_labels = [], []
        for batch_X, batch_y in dev_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            with torch.no_grad():
                outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f'Epoch {epoch + 1}, Dev Accuracy: {accuracy * 100:.2f}%, Dev F1 Score: {f1 * 100:.2f}%')

    # Evaluation on test set
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
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Test Accuracy: {accuracy * 100:.2f}%, Test F1 Score: {f1 * 100:.2f}%')

    return model


def train_alternative_classifier(X_train, X_dev, X_test, y_train, y_dev, y_test):
    scaler = StandardScaler()  # Initialize the scaler
    X_train = scaler.fit_transform(X_train)  # Scale the training data
    X_dev = scaler.transform(X_dev)  # Scale the development data
    X_test = scaler.transform(X_test)  # Scale the test data

    # SVM
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    svm_preds_dev = svm_classifier.predict(X_dev)
    svm_accuracy_dev = accuracy_score(y_dev, svm_preds_dev)
    svm_f1_dev = f1_score(y_dev, svm_preds_dev, average='weighted')
    print(f'SVM - Dev Accuracy: {svm_accuracy_dev * 100:.2f}%, Dev F1 Score: {svm_f1_dev * 100:.2f}%')

    # Random Forest
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    rf_preds_dev = rf_classifier.predict(X_dev)
    rf_accuracy_dev = accuracy_score(y_dev, rf_preds_dev)
    rf_f1_dev = f1_score(y_dev, rf_preds_dev, average='weighted')
    print(f'Random Forest - Dev Accuracy: {rf_accuracy_dev * 100:.2f}%, Dev F1 Score: {rf_f1_dev * 100:.2f}%')

    # Logistic Regression
    linear_classifier = LogisticRegression(max_iter=10000)  # Increased max_iter for convergence
    linear_classifier.fit(X_train, y_train)
    linear_preds_dev = linear_classifier.predict(X_dev)
    linear_accuracy_dev = accuracy_score(y_dev, linear_preds_dev)
    linear_f1_dev = f1_score(y_dev, linear_preds_dev, average='weighted')
    print(
        f'Linear Classifier - Dev Accuracy: {linear_accuracy_dev * 100:.2f}%, Dev F1 Score: {linear_f1_dev * 100:.2f}%')

    # XGBoost
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X_train, y_train)
    xgb_preds_dev = xgb_classifier.predict(X_dev)
    xgb_accuracy_dev = accuracy_score(y_dev, xgb_preds_dev)
    xgb_f1_dev = f1_score(y_dev, xgb_preds_dev, average='weighted')
    print(f'XGBoost - Dev Accuracy: {xgb_accuracy_dev * 100:.2f}%, Dev F1 Score: {xgb_f1_dev * 100:.2f}%')

    # Final evaluation on test set
    classifiers = [svm_classifier, rf_classifier, linear_classifier, xgb_classifier]
    classifier_names = ['SVM', 'Random Forest', 'Linear Classifier', 'XGBoost']

    for classifier, name in zip(classifiers, classifier_names):
        test_preds = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='weighted')
        print(f'{name} - Test Accuracy: {test_accuracy * 100:.2f}%, Test F1 Score: {test_f1 * 100:.2f}%')

if __name__ == "__main__":
    file_paths = {
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
