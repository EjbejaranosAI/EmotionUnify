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

    def mapping_sentiment(self, df):
        sentiment_mapping = {
            'neutral': 0,
            'positive': 1,
            'negative': 2
        }
        df['Sentiment_encoded'] = df['Sentiment'].map(sentiment_mapping)
        return df

    def extract_bert_features(self, df, batch_size=32):
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

    print("📂 Loading CSV Data... 📂")
    path_csv = "./train_sent_emo.csv"
    df_processed_train = pd.read_csv(path_csv)

    print("🤖 Extracting BERT Features... 🤖")
    text_feature_dict = feature_extractor.extract_bert_features(df_processed_train)

    print("💾 Saving Extracted Features... 💾")
    np.save('./text_features/train_text_features.npy', text_feature_dict)

    print("🎉 Feature Extraction Complete! 🎉")

    print("🔍 Inspecting the saved .npy file... 🔍")
    loaded_X_train = np.load('./text_features/train_text_features.npy', allow_pickle=True)
    print(f"📊 Shape: {loaded_X_train.shape}")
    print(f"🔢 Data Type: {loaded_X_train.dtype}")
    print(f"📈 Min Value: {np.min(loaded_X_train)}")
    print(f"📉 Max Value: {np.max(loaded_X_train)}")
    print("📝 File Inspection Complete 📝")

