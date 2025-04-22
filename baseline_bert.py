from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn
import pandas as pd

class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for idx in range(len(dataframe)):
            text = dataframe.iloc[idx]['text']
            label = dataframe.iloc[idx]['label']
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            })
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels

class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super().__init__()
        self.bert = pretrained_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled_output))

from utils import get_loss_fn, get_optimizer, train_model, get_validation_performance

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(bert_model, num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

data = pd.read_csv('combined_sarcasm_dataset.csv')
data = data.dropna(subset=['Content'])
dataset = pd.DataFrame({
    'text': data['Content'],
    'label': data['Label']
})

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

torch.manual_seed(42)
indices = torch.randperm(len(dataset)).tolist()
train_df = dataset.iloc[indices[:train_size]]
val_df = dataset.iloc[indices[train_size:train_size+val_size]]
test_df = dataset.iloc[indices[train_size+val_size:]]

train_loader = DataLoader(TextClassificationDataset(train_df, tokenizer), 
                         batch_size=16, collate_fn=collate_fn, shuffle=True)
val_loader = DataLoader(TextClassificationDataset(val_df, tokenizer), 
                       batch_size=16, collate_fn=collate_fn)

optimizer = get_optimizer(model, lr=2e-5, weight_decay=0.01)
best_model, stats = train_model(
    model, train_loader, val_loader, optimizer, num_epoch=15, device=device
)

test_loader = DataLoader(TextClassificationDataset(test_df, tokenizer), 
                       batch_size=16, collate_fn=collate_fn)
accuracy, loss = get_validation_performance(best_model, get_loss_fn(), test_loader, device)
print(f"Test Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
