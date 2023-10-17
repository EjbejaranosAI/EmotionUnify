import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel
from sklearn.metrics import f1_score, average_precision_score

# Function to load features from a pickle file
def load_features(file_path):
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    return features

# Load features
audio_features = load_features("../feature_extraction/audio_features/Dev_audio_features.pkl")
video_features = load_features("../feature_extraction/vision_features/Dev_video_features_crop_face.pkl")
text_features = load_features("../feature_extraction/text_features/dev_text_features.pkl")

# Find the common video_ids across all three dictionaries
common_video_ids = set(audio_features.keys()) & set(video_features.keys()) & set(text_features.keys())

# Align features based on common video_ids
aligned_audio_features = [torch.tensor(audio_features[vid]['features']) for vid in common_video_ids]
aligned_video_features = [torch.tensor(video_features[vid]['features']) for vid in common_video_ids]
aligned_text_features = [torch.tensor(text_features[vid]['features']) for vid in common_video_ids]
aligned_labels = [torch.tensor(audio_features[vid]['label']) for vid in common_video_ids]

# First step fusion network for Text and Audio
class TextAudioFusion(nn.Module):
    def __init__(self):
        super(TextAudioFusion, self).__init__()
        self.audio_projection = nn.Linear(1024, 256)
        self.text_projection = nn.Linear(768, 256)
        self.fusion_layer = nn.Linear(512, 256)

    def forward(self, audio, text):
        audio = self.audio_projection(audio)
        text = self.text_projection(text)
        concatenated = torch.cat((audio, text), dim=1)
        fused = self.fusion_layer(concatenated)
        return fused

# First step fusion network for Text and Video
class TextVideoFusion(nn.Module):
    def __init__(self):
        super(TextVideoFusion, self).__init__()
        self.video_projection = nn.Linear(512, 256)
        self.text_projection = nn.Linear(768, 256)
        self.fusion_layer = nn.Linear(512, 256)

    def forward(self, video, text):
        video = self.video_projection(video)
        text = self.text_projection(text)
        concatenated = torch.cat((video, text), dim=1)
        fused = self.fusion_layer(concatenated)
        return fused

# Second step fusion network for final prediction
class FinalFusion(nn.Module):
    def __init__(self):
        super(FinalFusion, self).__init__()
        self.final_layer = nn.Linear(512, 3)

    def forward(self, text_audio_fused, text_video_fused):
        concatenated = torch.cat((text_audio_fused, text_video_fused), dim=1)
        output = self.final_layer(concatenated)
        return output

# Initialize the models
text_audio_model = TextAudioFusion()
text_video_model = TextVideoFusion()
final_model = FinalFusion()



# Initialize the models and optimizer
text_audio_model = TextAudioFusion()
text_video_model = TextVideoFusion()
final_model = FinalFusion()

optimizer = optim.Adam([
    {'params': text_audio_model.parameters()},
    {'params': text_video_model.parameters()},
    {'params': final_model.parameters()}
], lr=0.001)

# Move to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
text_audio_model.to(device)
text_video_model.to(device)
final_model.to(device)

# Create PyTorch DataLoader
aligned_dataset = TensorDataset(torch.stack(aligned_audio_features), torch.stack(aligned_video_features), torch.stack(aligned_text_features), torch.stack(aligned_labels))
dataloader = DataLoader(aligned_dataset, batch_size=32, shuffle=True)

# Training Loop

f1_scores = []
f1_weighted_scores = []

# Training Loop
for epoch in range(150):  # Number of epochs
    all_preds = []
    all_labels = []

    for audio_batch, video_batch, text_batch, labels_batch in dataloader:
        # Move batches to the device
        audio_batch, video_batch, text_batch, labels_batch = audio_batch.to(device), video_batch.to(device), text_batch.to(device), labels_batch.to(device)

        # Forward pass through the first-step fusion models
        text_audio_fused = text_audio_model(audio_batch, text_batch)
        text_video_fused = text_video_model(video_batch, text_batch)

        # Forward pass through the second-step fusion model
        outputs = final_model(text_audio_fused, text_video_fused)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, torch.max(labels_batch, 1)[1])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics computation
        preds = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        labels = torch.max(labels_batch, 1)[1].cpu().detach().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    # Calculate the metrics after each epoch
    f1 = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    f1_scores.append(f1)
    f1_weighted_scores.append(f1_weighted)

    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}, F1-Score: {f1:.4f}, F1-Weighted: {f1_weighted:.4f}')