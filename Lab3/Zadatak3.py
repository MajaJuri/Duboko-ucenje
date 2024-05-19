import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import Zadatak1


class Model(nn.Module):
    def __init__(self,  args, input_size=300, output_size=1):
        super().__init__()
        self.num_layers = args['num_layers']
        self.hidden_size = args['hidden_size']
        if(args['prednauceneReprezentacije']):
            self.embedding = Zadatak1.embedding_layer
        else:
            self.embedding = Zadatak1.embedding_layer2
        self.cell_type = args['rnn_cell']
        self.dropout = args['dropout']
        self.bidirectional = args['bidirectional']
        self.dropout_layer = nn.Dropout(self.dropout)
        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional)
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional)
        else:# Vanilla
            self.rnn = nn.RNN(input_size, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
            self.fc2 = nn.Linear(self.hidden_size*2, output_size)
        else:
            self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.fc2 = nn.Linear(self.hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = x.float()
        x = self.dropout_layer(x)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(1), self.hidden_size)
        h0 = h0.float()
        if self.cell_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(1), self.hidden_size)
            c0 = c0.float()
            x, (hn, cn) = self.rnn(x, (h0, c0))  # samo LSTM ima celiju c
        else:
            x, hn = self.rnn(x, h0)
        x = x[-1, :, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(1)


def train(model, dataloader, optimizer, criterion, args):
    #print("Training model...")
    model.train()
    for (x, y, _) in dataloader:
        model.zero_grad()
        # RNN mreže preferiraju inpute u time-first formatu (budući da je brže iterirati po prvoj dimenziji tenzora)
        x = x.transpose(0, 1)  # transponirani input
        logits = model.forward(x)
        loss = criterion(logits, y.float())
        loss.backward()
        # gradient clipping prije optimizacijskog koraka
        torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
        optimizer.step()


def evaluate(model, dataloader, criterion, args):
    #print("Evaluating model...")
    model.eval()
    loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for (x, y_true, _) in dataloader:
            x = x.transpose(0, 1)  # Transpose inputs to time-first format
            logits = model.forward(x)
            y_true = y_true.float()
            loss += criterion(logits, y_true).item()
            preds = (logits > 0).float() # oni  koji su veci od 0 su 1, inace su 0
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

    _, train_dataloader, _, test_dataloader, _, valid_dataloader = Zadatak1.getData(
        batch_size_train=args['batch_size_train'], batch_size_test=args['batch_size_test_validate'], batch_size_valid=args['batch_size_test_validate'], shuffle=True)

    model = Model(args=args)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    for epoch in range(args['max_epochs']):
        print("Epoch {}:".format(epoch+1))
        train(model, train_dataloader, optimizer, criterion, args)
        loss, accuracy, f1, confusion = evaluate(model, valid_dataloader, criterion, args)
        print("\tvalid loss = {:.3f}\n\tvalid accuracy = {:.3f}%\n\tvalid F1 = {:.3f}\n".format(loss, accuracy * 100, f1))
        #print("Confusion matrix:")
        #print(confusion)
        #print()

    loss, accuracy, f1, confusion = evaluate(model, test_dataloader, criterion, args)
    print("\ttest loss = {:.3f}\n\ttest accuracy = {:.3f}%\n\ttest F1 = {:.3f}".format(loss, accuracy * 100, f1))
    print()


if __name__ == "__main__":
    seeds = [7052020, 1256437, 3452674, 9854326, 8543219]  # neki seedovi
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
            'clip': 0.25,
            'rnn_cell': 'LSTM',
            'num_layers': 2,
            'hidden_size': 150,
            'dropout': 0,
            'bidirectional': False,
            'prednauceneReprezentacije': True
        }
        main(args)


