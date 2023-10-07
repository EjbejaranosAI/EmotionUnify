import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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
    # Asegúrese de que y_train, y_dev, y_test sean tensores binarios one-hot
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_dev = torch.tensor(y_dev, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    # Prepare DataLoader for training, development, and test sets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), y_train)
    dev_dataset = TensorDataset(torch.tensor(X_dev, dtype=torch.float), y_dev)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Classifier(X_train.shape[1], y_train.shape[1]).to(device)  # Cambiado len(np.unique(y_train)) a y_train.shape[1]
    criterion = nn.BCEWithLogitsLoss()  # Cambiado de CrossEntropyLoss a BCEWithLogitsLoss
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