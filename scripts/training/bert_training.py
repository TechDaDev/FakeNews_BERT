import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from datetime import datetime
import json

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from src.bert_utils import BERTDataset, get_bert_model, load_bert_tokenizer

def train_bert():
    # Configuration
    MODEL_NAME = 'distilbert-base-uncased' # Using DistilBERT for efficiency
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LR = 2e-5
    DATA_PATH = root_dir / 'data' / 'WELFake_Dataset.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    data = pd.read_csv(DATA_PATH)
    # Basic cleaning
    data = data.dropna(subset=['text', 'label'])
    
    # WELFake mapping (0=Real, 1=Fake)
    data['label'] = data['label'].astype(int)

    # Use a subset if you are on CPU or want faster training for testing
    # data = data.sample(2000, random_state=42) 

    texts = data['text'].values
    labels = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 2. Tokenizer & Datasets
    tokenizer = load_bert_tokenizer(MODEL_NAME)
    
    train_dataset = BERTDataset(X_train, y_train, tokenizer, MAX_LEN)
    test_dataset = BERTDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. Model
    model = get_bert_model(MODEL_NAME, num_labels=2)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # 5. Training Loop
    print("Starting BERT Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 6. Evaluation
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))

    # 7. Save Artifacts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = root_dir / 'saved_models' / 'model_runs' / f'BERT_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save results
    results = {
        'model_name': MODEL_NAME,
        'accuracy': acc,
        'epoch': EPOCHS,
        'batch_size': BATCH_SIZE,
        'timestamp': timestamp
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Model and tokenizer saved to {save_dir}")

if __name__ == "__main__":
    train_bert()
