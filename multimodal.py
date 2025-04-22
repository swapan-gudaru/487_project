import torch
import pandas as pd
import emoji
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import gensim.models as gsm

e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)

def process_emojis(text):
    """Extract and convert emojis to vectors using emoji2vec"""
    emojis = [c for c in text if c in emoji.EMOJI_DATA]
    
    vectors = []
    for e in emojis:
        if e in e2v.key_to_index:
            vectors.append(e2v[e])
    
    if not vectors:
        return torch.zeros(300)
    
    avg_vector = np.mean(vectors, axis=0)
    return torch.tensor(avg_vector, dtype=torch.float32)

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        emoji_vector = process_emojis(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emoji_features': emoji_vector,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

class MultimodalBERT(torch.nn.Module):
    def __init__(self, bert_model, hidden_dim=256, num_labels=2):
        super().__init__()
        self.bert = bert_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768 + 300, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, emoji_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat((pooled_output, emoji_features), dim=1)
        return self.classifier(combined)

def train():
    comments = pd.read_csv('facebook_comments.csv')['Comments'].tolist()
    labels = pd.read_csv('facebook_labels.csv').values.flatten().tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        comments, labels, test_size=0.15, random_state=42)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model = MultimodalBERT(bert_model)
    
    train_loader = DataLoader(
        SarcasmDataset(train_texts, train_labels, tokenizer),
        batch_size=64, shuffle=True)
    
    val_loader = DataLoader(
        SarcasmDataset(val_texts, val_labels, tokenizer),
        batch_size=64)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(15):
        model.train()
        for batch in train_loader:
            inputs = {k: v.to(device) for k, v in batch.items() 
                     if k in ['input_ids', 'attention_mask']}
            inputs['emoji_features'] = batch['emoji_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() 
                         if k in ['input_ids', 'attention_mask']}
                inputs['emoji_features'] = batch['emoji_features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(**inputs)
                val_loss += loss_fn(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
        
        print(f"Epoch {epoch+1}: Val Loss {val_loss/len(val_loader):.4f}, "
              f"Accuracy {correct/len(val_labels):.4f}")

    torch.save(model.state_dict(), 'sarcasm_detector.pth')

if __name__ == "__main__":
    train()
