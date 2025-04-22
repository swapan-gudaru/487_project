# Multimodal Sarcasm Detection

This project implements a multimodal sarcasm detection system that leverages both text and emoji features to classify social media comments as sarcastic or not sarcastic. The repository includes data preprocessing, traditional machine learning, BERT-based, and multimodal deep learning approaches.

## Workflow

### 1. Data Preprocessing

Run the preprocessing script to generate the combined dataset:

```
python Preprocessing.py
```

- **Input:** Raw data files (`facebook_comments.csv`, `facebook_labels.csv`, `twitter_comments.csv`, `twitter_labels.csv`, `reddit.csv`, `sarcasm.csv`)
- **Output:** `combined_sarcasm_dataset.csv`

---

### 2. Logistic Regression Baseline

Train and evaluate a logistic regression model using TF-IDF features:

```
python logistic.py
```

---

### 3. BERT Baseline Model

Train and evaluate a BERT-based sarcasm classifier:

```
python baseline_bert.py
```

- **Note:** Requires `utils.py` in the same directory.

---

### 4. Multimodal BERT Model (Text + Emoji)

Train and evaluate the multimodal BERT model that incorporates emoji embeddings:

```
python multimodal.py
```

- **Note:** Requires `emoji2vec.bin` and `utils.py` in the same directory.

---

## Files Overview

- `Preprocessing.py`: Cleans and merges raw data into a unified dataset.
- `logistic.py`: Implements the TF-IDF + Logistic Regression baseline.
- `baseline_bert.py`: Implements the BERT-based sarcasm detection baseline.
- `multimodal.py`: Implements the multimodal model (BERT + emoji2vec).
- `utils.py`: Utility functions for model training and evaluation.
- `emoji2vec.bin`: Pre-trained emoji embeddings.
- `combined_sarcasm_dataset.csv`: Output dataset for model training/testing.
