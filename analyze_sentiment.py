import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
file_name = 'C:/Users/Argon Aliu/Downloads/New folder (9)/Sentiment Analysis Model/sentences1.csv'
data = pd.read_csv(file_name)

# Function to clean up special characters in text
def clean_text(text):
    text = text.replace('â€œ', '"').replace('â€™', "'").replace('â€˜', "'").replace('â€”', '—')
    return text

data['sentence'] = data['sentence'].apply(clean_text)

# Check for non-finite values in the label column
if data['final_sentiment'].isnull().any():
    data['final_sentiment'].fillna('neutral', inplace=True)

# Convert categorical labels to numeric
label_mapping = {'negative': 0, 'positive': 1, 'neutral': 2}
data['numeric_label'] = data['final_sentiment'].map(label_mapping)
data['numeric_label'] = data['numeric_label'].astype(int)

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2)

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
        return item

    def __len__(self):
        return len(self.labels)

# Preprocess the data
def preprocess_data(data):
    texts = data['sentence'].tolist()
    labels = data['numeric_label'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(texts, truncation=True, padding=True)
    return encodings, labels

# Process both training and validation datasets
train_encodings, train_labels = preprocess_data(train_data)
val_encodings, val_labels = preprocess_data(val_data)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_mapping))
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    dataloader_pin_memory=False,
)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train and evaluate the model
trainer.train()
evaluation_results = trainer.evaluate()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_bert_model')

print("Model fine-tuning and evaluation complete.")
