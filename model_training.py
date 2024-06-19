from data_processing import *
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report

merged_df['target'] = np.random.randint(0, 2, merged_df.shape[0])

# Define a custom dataset
class LyricsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define parameters
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3

# Prepare dataset
dataset = LyricsDataset(
    texts=merged_df['processed_lyrics'].tolist(),
    labels=merged_df['target'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['label'].to(model.device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Training loss: {avg_train_loss}')

    # Validation loop
    model.eval()
    val_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation loss: {avg_val_loss}')
    print(f'Validation Accuracy: {accuracy_score(true_labels, predictions)}')

# Evaluate the model
print("Classification Report:\n", classification_report(true_labels, predictions))