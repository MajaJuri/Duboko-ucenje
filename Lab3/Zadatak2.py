import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import Zadatak1


class BasicModel(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=150, prednauceneReprezentacije=True):
        super().__init__()
        self.prednauceneReprezentacije = prednauceneReprezentacije # freeze
        self.avg_pool = nn.AdaptiveAvgPool1d(1)# uzme cijeli polje i vrati jednu vrijednost
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x): # [batch_size, T]
        if self.prednauceneReprezentacije:
            x = Zadatak1.embedding_layer(x) # [batch_size, T, input_dim]
        else:
            x = Zadatak1.embedding_layer2(x)
        x = x.permute(0, 2, 1)  # [batch_size, input_dim, T]
        x = self.avg_pool(x)  # [batch_size, input_dim, 1]
        x = x.view(x.size(0), -1)
        x = self.fc1(x.float())  # [batch_size, hidden_dim]
        x = self.relu1(x)
        x = self.fc2(x)  # [batch_size, hidden_dim]
        x = self.relu2(x)
        x = self.fc3(x)  # [batch_size, 1]
        return x.squeeze(1)


def train(model, dataloader, optimizer, criterion):
    # print("\tTraining model...")
    model.train()
    for (x, y_true, _) in dataloader:
        model.zero_grad()
        logits = model.forward(x)
        loss = criterion(logits, y_true.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, criterion):
    # print("\tEvaluating model...")
    model.eval()
    loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for (x, y_true, _) in dataloader:
            logits = model.forward(x)
            y_true = y_true.float()
            loss += criterion(logits, y_true).item()
            preds = (logits > 0).float()
            correct += (preds == y_true).sum().item()
            all_labels.extend(y_true.tolist())
            all_preds.extend(preds.tolist())
    accuracy = correct / len(all_labels)
    f1 = f1_score(all_labels, all_preds)
    confusion = confusion_matrix(all_labels, all_preds)
    loss = loss / len(dataloader)
    return loss, accuracy, f1, confusion


def main(args):
    torch.manual_seed(seed=args['seed'])
    np.random.seed(seed=args['seed'])

    train_dataset, dataloader_train, test_dataset, dataloader_test, valid_dataset, dataloader_valid = Zadatak1. \
        getData(batch_size_train=args['batch_size_train'], batch_size_test=args['batch_size_test_validate'],
                batch_size_valid=args['batch_size_test_validate'])

    model = BasicModel(prednauceneReprezentacije=args['prednauceneReprezentacije'])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    # treniranje
    for epoch in range(args['max_epochs']):
        train(model, dataloader_train, optimizer, criterion)
        loss, accuracy, f1, confusion = evaluate(model, dataloader_valid, criterion)
        print("\tEpoch {}:\n\t\tvalid loss = {:.3f}\n\t\tvalid accuracy = {:.3f}%\n\t\tvalid F1 = {:.3f}".format(epoch + 1, loss, accuracy * 100, f1))
        #print("Confusion matrix:")
        #print(confusion)
        #print()

    # evaluacija modela
    loss, accuracy, f1, confusion = evaluate(model, dataloader_test, criterion)
    print("\tTest loss = {:.3f}\n\ttest accuracy = {:.3f}%\n\ttest F1 = {:.3f}".format(loss, accuracy * 100, f1))
    print()


if __name__ == "__main__":

    seeds = [7052020, 1256437, 3452674, 9854326, 8543219] # neki seedovi

    for (i, seed) in enumerate(seeds):
        print("Seed {}/{}: {}".format(i+1, len(seeds), seed))
        args = {
            'seed': seed,
            'min_freq': 1,
            'max_size': -1,
            'learning_rate': 1e-4,
            'batch_size_train': 10,
            'batch_size_test_validate': 32,
            'max_epochs': 5,
            'prednauceneReprezentacije': True
        }
        main(args)

