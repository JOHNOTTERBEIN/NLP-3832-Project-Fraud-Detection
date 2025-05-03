#!/usr/bin/env python3
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm

# Hyperparameters
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
MAX_LEN    = 128
EPOCHS     = 2
LR         = 2e-5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed splits
train_df = pd.read_csv("/home/joot9454/fraud_detection/data/train.csv")
val_df   = pd.read_csv("/home/joot9454/fraud_detection/data/val.csv")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset wrapper
class JobDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts.tolist()
        self.labels = labels.tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = JobDataset(train_df["text"], train_df["label"])
val_ds   = JobDataset(val_df["text"],   val_df["label"])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# Model, optimizer, scheduler
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} train loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(preds)
    print(f"Epoch {epoch+1} val accuracy: {correct/total:.4f}")

print("Training complete!")

# Save:
save_dir = "/home/joot9454/fraud_detection/bert_finetuned"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model & tokenizer saved to {save_dir}/")
