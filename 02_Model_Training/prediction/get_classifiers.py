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
from sklearn.metrics import average_precision_score


from utils.utils_data import load_features, prepare_data, merge_modalities
from utils.evaluation_classifier import evaluate_classifier, calculate_map


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_classifier(X_train, X_dev, X_test, y_train, y_dev, y_test):
    # Prepare DataLoader for training, development, and test sets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long))
    dev_dataset = TensorDataset(torch.tensor(X_dev, dtype=torch.float), torch.tensor(y_dev, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
        all_probs, all_labels = [], []
        for batch_X, batch_y in dev_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            with torch.no_grad():
                outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Obtenga las probabilidades
            all_probs.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())

        # Asegúrate de pasar all_probs en lugar de all_preds
        dev_map = calculate_map(all_labels, all_probs)
        print(f'Epoch {epoch + 1}, validation MaP: {dev_map * 100:.2f}%')

    # Evaluation on test set
    model.eval()
    all_probs, all_labels = [], []
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        with torch.no_grad():
            outputs = model(batch_X)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Obtenga las probabilidades
        all_probs.extend(probs)
        all_labels.extend(batch_y.cpu().numpy())

    # Suponiendo que las etiquetas están en formato binario one-hot
    test_map = calculate_map(all_labels, all_probs)
    print(f'Test MaP: {test_map * 100:.2f}%')

    return model



def train_alternative_classifier(X_train, X_dev, X_test, y_train, y_dev, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    classifiers = {
        'SVM': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Linear Classifier': LogisticRegression(max_iter=10000),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, classifier in classifiers.items():
        print(f'Training {name}...')
        classifier.fit(X_train, y_train)
        print(f'Evaluating {name} on dev set...')
        evaluate_classifier(classifier, X_dev, y_dev, name)

    print('Final evaluation on test set...')
    for name, classifier in classifiers.items():
        evaluate_classifier(classifier, X_test, y_test, name)


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
