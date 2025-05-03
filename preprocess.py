#!/usr/bin/env python3
import pandas as pd
import re, string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# 1) Load & drop unwanted columns
job_df = pd.read_csv("/home/joot9454/fraud_detection/data/fake_job_postings.csv")
if 'Unnamed: 0' in job_df.columns:
    job_df = job_df.drop(columns=['Unnamed: 0'])

# 2) Drop rows with any missing values
job_df = job_df.dropna()

# 3) Assemble a clean DataFrame combining all text fields
job_df_clean = pd.DataFrame()
job_df_clean["text"] = (
    job_df
    .drop(columns=["fraudulent"])
    .astype(str)
    .agg(" ".join, axis=1)
)
# Map your labels exactly as in the notebook
if job_df["fraudulent"].dtype == object:
    job_df_clean["fraudulent"] = job_df["fraudulent"].map({"Fake": 1, "Real": 0})
else:
    job_df_clean["fraudulent"] = job_df["fraudulent"].astype(int)

# 4) Define your removal-of-stopwords-and-punctuation function
stop_words = set(stopwords.words("english"))
def preprocess_remove_sw_punc(text):
    text = text.lower()
    text = "".join(ch for ch in text if not ch.isdigit())
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # remove stopwords
    text = " ".join(w for w in text.split() if w not in stop_words)
    # collapse extra spaces
    return " ".join(text.split())

# 5) Apply cleaning
job_df_clean_sw = job_df_clean.copy()
job_df_clean_sw["text"] = job_df_clean["text"].apply(preprocess_remove_sw_punc)

# 6) Stratified 80/10/10 split
X = job_df_clean_sw["text"]
y = job_df_clean_sw["fraudulent"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 7) Save to CSV for downstream training
pd.DataFrame({"text": X_train, "label": y_train}).to_csv("/home/joot9454/fraud_detection/data/train.csv", index=False)
pd.DataFrame({"text": X_val,   "label": y_val  }).to_csv("/home/joot9454/fraud_detection/data/val.csv",   index=False)
pd.DataFrame({"text": X_test,  "label": y_test }).to_csv("/home/joot9454/fraud_detection/data/test.csv",  index=False)

print("Preprocessing complete!")
print("Train / Val / Test sizes:", len(X_train), len(X_val), len(X_test))

