#!/usr/bin/env python3
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# 1) Configuration
MODEL_DIR = "/home/joot9454/fraud_detection/bert_finetuned"   # or wherever you saved your model
TOKENIZER_NAME = "bert-base-uncased"
TEST_CSV = "/home/joot9454/fraud_detection/data/test.csv"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# 3) Prepare test Dataset
class FraudTestDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts.tolist()
        self.labels = labels.tolist()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":     enc["input_ids"].squeeze(0),
            "attention_mask":enc["attention_mask"].squeeze(0),
            "label":         torch.tensor(self.labels[idx], dtype=torch.long)
        }

test_df = pd.read_csv(TEST_CSV)
test_ds = FraudTestDataset(test_df["text"], test_df["label"])
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# 4) Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids     = batch["input_ids"].to(DEVICE)
        attention_mask= batch["attention_mask"].to(DEVICE)
        labels        = batch["label"].to(DEVICE)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds .extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# 5) Metrics
print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["Real","Fake"]))

cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print(f"\nFalse Positives (Real→Fake): {fp}")
print(f"False Negatives (Fake→Real): {fn}")

# 6) Error analysis discussion
print("\n--- Real-World Consequences ---")
print(f"• False Positives: {fp} legitimate job posts flagged as fraudulent. "
      "This risks filtering out real opportunities and eroding user trust.")
print(f"• False Negatives: {fn} fraudulent posts slip through. "
      "This leaves consumers exposed to scams, undermining platform safety.")

