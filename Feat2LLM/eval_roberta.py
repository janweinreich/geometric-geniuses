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
import numpy as np
import argparse
import os

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

# Load data from JSON
def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--small', type=bool, help='if true, run on small molecules')
    parser.add_argument('--data', type=str, help='name of dataset to use')
    # options for representation cMBDF, cMBDF_trans, (SPAHM, SPAHM_trans)
    parser.add_argument('--rep', type=str, help='name of representation to use')
    parser.add_argument(
        "--modal", type=str, help="name of modality to use", default="vec"
    )
    args = parser.parse_args()

    test_data = load_data("{}_{}_test.json".format(args.data, args.rep))
    #_, test_data = train_test_split(data, test_size=0.95, random_state=42)  # Assuming you use the same split
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_dataset = MoleculeDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the model
    # Set device: Apple/NVIDIA/CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = RobertaForRegression().to(device)
    checkpoint = torch.load("save_models/{}_{}_regression.pth".format(args.data, args.rep))
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
    # mean squared error
    mse = np.mean((np.array(predictions) - np.array(actuals))**2)
    print(f'Mean Squared Error: {mse}')
    # mean absolute error
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    print(f'Mean Absolute Error: {mae}')

    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, color='blue', alpha=0.5)

    min_value = min(np.min(actuals), np.min(predictions))
    max_value = max(np.max(actuals), np.max(predictions))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Energy')
    plt.ylabel('Predicted Energy')
    plt.grid(True)  # Enable grid lines for better readability
    plt.xticks(fontsize=16)  # Adjust x-axis tick label font size
    plt.yticks(fontsize=16)  # Adjust y-axis tick label font size
    # create subfolder for figures if it does not exist
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.savefig('figures/{}_{}_regression.png'.format(args.data, args.rep))

    plt.show()

    # create a text file with all the metrics
    with open("figures/{}_{}_regression.txt".format(args.data, args.rep), "w") as f:
        f.write(f"R² Score: {r2}\n")
        f.write(f"Mean Squared Error: {mse}\n")
        f.write(f"Mean Absolute Error: {mae}\n")

    # Save the predictions and actual values
    with open("figures/{}_{}_regression_predictions.txt".format(args.data, args.rep), "w") as f:
        f.write("Actual, Predicted\n")
        for actual, predicted in zip(actuals, predictions):
            f.write(f"{actual}, {predicted}\n")
