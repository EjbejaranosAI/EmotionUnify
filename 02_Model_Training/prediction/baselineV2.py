import pickle
import torch
import matplotlib.pyplot as plt
import configparser
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import random_split

class FeatureLoader:
    @staticmethod
    def load_features(file_path):
        with open(file_path, 'rb') as f:
            features = pickle.load(f)
        return features

class FusionModels:
    class FinalFusion(nn.Module):
        def __init__(self):
            super(FusionModels.FinalFusion, self).__init__()
            self.final_layer = nn.Linear(512, 3)

        def forward(self, text_audio_fused, text_video_fused):
            concatenated = torch.cat((text_audio_fused, text_video_fused), dim=1)
            output = self.final_layer(concatenated)
            return output

    class TextAudioFusion(nn.Module):
        def __init__(self):
            super(FusionModels.TextAudioFusion, self).__init__()
            self.audio_projection = nn.Linear(1024, 256)
            self.text_projection = nn.Linear(768, 256)
            self.fusion_layer = nn.Linear(512, 256)
            self.aditional_layer = nn.Linear(256,256)
            self.dropout = nn.Dropout(0.5)

        def forward(self, audio, text):
            audio = self.audio_projection(audio)
            text = self.text_projection(text)
            concatenated = torch.cat((audio, text), dim=1)
            fused = self.fusion_layer(concatenated)
            fused = self.dropout(fused)
            return fused


    class TextVideoFusion(nn.Module):
        def __init__(self):
            super(FusionModels.TextVideoFusion, self).__init__()
            self.video_projection = nn.Linear(512, 256)
            self.text_projection = nn.Linear(768, 256)
            self.fusion_layer = nn.Linear(512, 256)
            self.additional_layer = nn.Linear(256, 256)
            self.dropout = nn.Dropout(0.1)

        def forward(self, video, text):
            video = self.video_projection(video)
            text = self.text_projection(text)
            concatenated = torch.cat((video, text), dim=1)
            fused = self.fusion_layer(concatenated)
            fused = self.additional_layer(fused)
            fused = self.dropout(fused)
            return fused


class Trainer:
    def __init__(self, train_dataloader, val_dataloader, device, early_stopping_patience=80):
        self.text_audio_model = FusionModels.TextAudioFusion()
        self.text_video_model = FusionModels.TextVideoFusion()
        self.final_model = FusionModels.FinalFusion()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = early_stopping_patience

        # Adding weight_decay for L2 regularization
        self.optimizer = optim.Adam([
            {'params': self.text_audio_model.parameters()},
            {'params': self.text_video_model.parameters()},
            {'params': self.final_model.parameters()}
        ], lr=0.001, weight_decay=1e-4)


    def plot_metrics(self, train_metrics, val_metrics, metric_name):
        plt.plot(train_metrics, label=f'Training {metric_name}')
        plt.plot(val_metrics, label=f'Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plt.show()

    def save_model(self, epoch, path='./'):
        torch.save(self.final_model.state_dict(), f'{path}/final_model_best.pth')
        torch.save(self.text_audio_model.state_dict(), f'{path}/text_audio_model_best.pth')
        torch.save(self.text_video_model.state_dict(), f'{path}/text_video_model_best.pth')

    def train(self, epochs=500):
        train_f1_scores = []
        val_f1_scores = []

        for epoch in range(epochs):
            self.final_model.train()
            self.text_audio_model.train()
            self.text_video_model.train()
            all_train_preds = []
            all_train_labels = []
            all_val_preds = []
            all_val_labels = []
            train_loss = 0

            # Training Loop
            for audio_batch, video_batch, text_batch, labels_batch in self.train_dataloader:
                audio_batch, video_batch, text_batch, labels_batch = audio_batch.to(self.device), video_batch.to(
                    self.device), text_batch.to(self.device), labels_batch.to(self.device)

                text_audio_fused = self.text_audio_model(audio_batch, text_batch)
                text_video_fused = self.text_video_model(video_batch, text_batch)
                outputs = self.final_model(text_audio_fused, text_video_fused)

                criterion = nn.CrossEntropyLoss()
                loss = criterion(outputs, torch.max(labels_batch, 1)[1])
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                labels = torch.max(labels_batch, 1)[1].cpu().detach().numpy()

                all_train_preds.extend(preds)
                all_train_labels.extend(labels)

            # Validation Loop
            val_loss = 0
            self.final_model.eval()
            self.text_audio_model.eval()
            self.text_video_model.eval()
            with torch.no_grad():
                for audio_batch, video_batch, text_batch, labels_batch in self.val_dataloader:
                    audio_batch, video_batch, text_batch, labels_batch = audio_batch.to(self.device), video_batch.to(
                        self.device), text_batch.to(self.device), labels_batch.to(self.device)

                    text_audio_fused = self.text_audio_model(audio_batch, text_batch)
                    text_video_fused = self.text_video_model(video_batch, text_batch)

                    outputs = self.final_model(text_audio_fused, text_video_fused)

                    loss = criterion(outputs, torch.max(labels_batch, 1)[1])
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                    labels = torch.max(labels_batch, 1)[1].cpu().detach().numpy()

                    all_val_preds.extend(preds)
                    all_val_labels.extend(labels)

            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model(epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
            train_f1_scores.append(train_f1)

            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
            val_f1_scores.append(val_f1)

            print(f'Epoch [{epoch + 1}/{epochs}], Train F1-Score: {train_f1:.4f}, Val F1-Score: {val_f1:.4f}')

        self.plot_metrics(train_f1_scores, val_f1_scores, 'F1-Score')

def main():
    config_file = "./../config.ini"
    config = configparser.ConfigParser()
    config.read(config_file)

    audio_features = FeatureLoader.load_features(config['General'].get('audio_features'))
    video_features = FeatureLoader.load_features(config['General'].get('video_features'))
    text_features = FeatureLoader.load_features(config['General'].get('text_features'))

    common_video_ids = set(audio_features.keys()) & set(video_features.keys()) & set(text_features.keys())

    aligned_audio_features = [torch.tensor(audio_features[vid]['features']) for vid in common_video_ids]
    aligned_video_features = [torch.tensor(video_features[vid]['features']) for vid in common_video_ids]
    aligned_text_features = [torch.tensor(text_features[vid]['features']) for vid in common_video_ids]
    aligned_labels = [torch.tensor(audio_features[vid]['label']) for vid in common_video_ids]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_dataset = TensorDataset(
        torch.stack(aligned_audio_features),
        torch.stack(aligned_video_features),
        torch.stack(aligned_text_features),
        torch.stack(aligned_labels)
    )

    # Calculate the sizes for training and validation sets (80/20 split)
    total_size = len(full_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoader instances for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Instantiate the Trainer class with the required arguments
    trainer = Trainer(train_dataloader, val_dataloader, device)
    trainer.train()

if __name__ == "__main__":
    main()