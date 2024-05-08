import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
import pdb
# Load data from JSON file
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Assuming the filepath to your JSON file
data = load_data('qm7_train_smi.json')

# Split the data into training and test sets (modify as needed if already split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

class MoleculeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.encodings = tokenizer([d['input'] for d in data], truncation=True, padding=True, max_length=512)
        self.labels = [float(d['output']) for d in data]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_dataset = MoleculeDataset(train_data, tokenizer)
test_dataset = MoleculeDataset(test_data, tokenizer)

# Define the custom model with a regression head
class RobertaForRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.regression_head = nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        logits = self.regression_head(sequence_output)
        return logits

# Set device: Apple/NVIDIA/CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = RobertaForRegression().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
model.train()
for epoch in range(30):  # Number of epochs
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)
        outputs = model(inputs, mask).squeeze(-1)
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)
        outputs = model(inputs, mask).squeeze(-1)
        loss = nn.MSELoss()(outputs, labels)
        total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader)}")


import os

# Save model and optimizer state
def save_model(model, optimizer, epoch, loss, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, filepath)

# Assuming you want to save the model after training
model.eval()
save_model(model, optimizer, epoch, loss.item(), 'roberta_regression.pth')