import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel


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



# Load data from JSON
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

data = load_data('qm7_test_smi.json')
_, test_data = train_test_split(data, test_size=0.2, random_state=42)  # Assuming you use the same split

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
test_dataset = MoleculeDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the model
device = torch.device("cpu")
model = RobertaForRegression().to(device)
checkpoint = torch.load('roberta_regression.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate the model
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        mask = batch['attention_mask'].to(device)
        outputs = model(inputs, mask).squeeze(-1)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

# Compute R² score
r2 = r2_score(actuals, predictions)
print(f'R² Score: {r2}')

# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, color='blue', alpha=0.5)
plt.title('Actual vs. Predicted Solvation Energies')
plt.xlabel('Actual Energy')
plt.ylabel('Predicted Energy')
plt.grid(True)
plt.savefig('actual_vs_predicted.png')
plt.show()

