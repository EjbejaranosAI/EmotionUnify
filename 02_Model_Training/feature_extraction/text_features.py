import configparser
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import time
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class TextFeatureExtractor:
    def __init__(self, classification_type="Emotion"):
        config = configparser.ConfigParser()
        config.read('../config.ini')
        self.model_name = config.get('TextFeatureExtraction', 'model_name')
        self.max_length = config.getint('TextFeatureExtraction', 'max_length')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.classification_type = classification_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def tokenize_utterances(self, utterances):
        input_ids = []
        attention_masks = []
        for utterance in tqdm(utterances, desc="Tokenizing Utterances"):
            tokens = self.tokenizer.encode(
                utterance,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            )
            input_ids.append(tokens)
            attention_masks.append([1] * len(tokens))
        return torch.tensor(input_ids), torch.tensor(attention_masks)

    def mapping_emotion(self, df):
        emotion_mapping = {
            'neutral': [1, 0, 0, 0, 0, 0, 0],
            'surprise': [0, 1, 0, 0, 0, 0, 0],
            'fear': [0, 0, 1, 0, 0, 0, 0],
            'sadness': [0, 0, 0, 1, 0, 0, 0],
            'joy': [0, 0, 0, 0, 1, 0, 0],
            'disgust': [0, 0, 0, 0, 0, 1, 0],
            'anger': [0, 0, 0, 0, 0, 0, 1]
        }
        df['Emotion_encoded'] = df['Emotion'].map(emotion_mapping)
        return df

    def save_features(self, text_feature_dict, save_path):
        print(f"💾 Saving Extracted Features to {save_path}... 💾")
        np.save(save_path, text_feature_dict)

    def mapping_sentiment(df):
        sentiment_mapping = {
            'neutral': [1, 0, 0],
            'positive': [0, 1, 0],
            'negative': [0, 0, 1]
        }
        df['Sentiment_encoded'] = df['Sentiment'].map(sentiment_mapping)
        return df

    def extract_bert_features(self, file_path, batch_size=32):
        df = pd.read_csv(file_path)
        if self.classification_type == "Emotion":
            df = self.mapping_emotion(df)
        elif self.classification_type == "Sentiment":
            df = self.mapping_sentiment(df)

        utterances = df['transcription'].tolist()
        input_ids, attention_masks = self.tokenize_utterances(utterances)

        # Convert to tensor and move to the specified device
        input_ids, attention_masks = input_ids.to(self.device), attention_masks.to(self.device)

        print("⏳ Starting Feature Extraction... ⏳")
        start_time = time.time()

        # Create a DataLoader to handle batching of data
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Initializing empty lists to store the outputs
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Features"):
                input_ids_batch, attention_masks_batch = batch
                bert_outputs = self.model(input_ids_batch, attention_mask=attention_masks_batch)
                all_outputs.append(bert_outputs[0][:, 0, :].cpu().numpy())  # Moving tensors to cpu

        # Concatenating the results from all batches
        X = np.concatenate(all_outputs, axis=0)

        # Assuming the labels are numeric
        y = df[f"{self.classification_type}_encoded"].values

        end_time = time.time()
        print(f"⌛ Feature Extraction completed in {end_time - start_time:.2f} seconds ⌛")

        # Creating a dictionary to store both features and labels
        text_feature_dict = {f"text_{index}": {'features': features, 'label': label}
                             for index, (features, label) in enumerate(zip(X, y))}

        return text_feature_dict


if __name__ == "__main__":
    print("🚀 Initializing Text Feature Extraction for Emotion Classification... 🚀")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = TextFeatureExtractor("Sentiment")

    datasets = {
        'train': './text_features/train_sent_emo.csv',
        'dev': './text_features/dev_sent_emo.csv',
        'test': './text_features/test_sent_emo.csv'
    }

    for split, path_csv in datasets.items():
        print(f"📂 Loading {split.upper()} CSV Data... 📂")
        print(f"🤖 Extracting BERT Features for {split.upper()} set... 🤖")
        text_feature_dict = feature_extractor.extract_bert_features(path_csv)

        save_path = f'./text_features/{split}_text_features.npy'
        feature_extractor.save_features(text_feature_dict, save_path)

        print(f"🎉 {split.upper()} Feature Extraction Complete! 🎉")

        print(f"🔍 Inspecting the saved .npy file for {split.upper()} set... 🔍")
        loaded_features = np.load(save_path, allow_pickle=True)
        print(f"📊 Shape: {loaded_features.shape}")
        print(f"🔢 Data Type: {loaded_features.dtype}")
        print(f"📈 Min Value: {np.min(loaded_features)}")
        print(f"📉 Max Value: {np.max(loaded_features)}")
        print(f"📝 {split.upper()} File Inspection Complete 📝")


