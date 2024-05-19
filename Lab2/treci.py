import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pathlib import Path
import numpy as np
import math
import skimage as ski
import os
import matplotlib.pyplot as plt


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def draw_conv_filters(epoch, tensor, save_dir, dodatno):
  C = 10
  w = tensor.weight.detach().cpu().numpy()
  num_filters = w.shape[0]
  k = tensor.kernel_size[0] #k = int(np.sqrt(w.shape[1] / C)) # w.shape[-1] int(np.sqrt(w.shape[1] / C))
  #w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  #for i in range(C):
  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = '%s_epoch_%02d_%s.png' % ('conv1', epoch, dodatno)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def nacrtaj_i_spremi_graf(x_os, y_os, save_dir, label="", x_label="", y_label=""):
    plt.plot(x_os, y_os, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    fig_title = save_dir / 'graf.png'
    plt.savefig(fig_title)


def prebrojiTocne(y_pred, y_true):
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    cnt_correct = np.sum(y_pred_np == y_true_np)
    return cnt_correct


class Net(nn.Module):
    def __init__(self, weight_decay):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()
        self.flatten3 = nn.Flatten()
        self.fc3 = nn.Linear(32 * 4 * 4, 512) # 32*7*7
        self.relu3 = nn.ReLU()
        self.logits = nn.Linear(512, 10)
        #self.regularizer = nn.Linear(1, 1, bias=False)
        self.weight_decay = weight_decay

    def forward(self, x):
        #print(x.numpy().shape)
        x = self.conv1(x.float())
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.flatten3(x)
        x = self.fc3(x)
        x = self.relu3(x)
        y = self.logits(x)
        return y


def train(train_dataloader, valid_dataloader, model, config):
    save_dir = config['save_dir']
    max_epochs = config['max_epochs']
    weight_decay = config['weight_decay']
    learning_rate = config['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss_array = []
    train_accuracy_array = []
    validation_accuracy_array = []
    validation_loss_array = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_epoch_correct = 0
        total_epoch_loss = 0
        for i, (x, y_true) in enumerate(train_dataloader):
            outputs = model.forward(x)
            #print("outputs", outputs)

            loss = nn.CrossEntropyLoss()(outputs, y_true)

            optimizer.zero_grad()
            loss.backward()
            total_epoch_loss = total_epoch_loss + loss.item()
            optimizer.step()

            _, y_pred = outputs.max(1)

            # izracunaj broj tocnih
            total_epoch_correct = total_epoch_correct + prebrojiTocne(y_pred, y_true)
            # print(total_epoch_correct)

            # za svaki stoti batch ispisati loss
            if i % 100 == 0 and i > 0:
                print('epoch %d, step %d, item loss = %.2f' %
                      (epoch, i, loss.item()))

        # izracunati accuracy i tocnost
        train_accuracy = (total_epoch_correct / len(train_dataloader.dataset))
        average_loss = total_epoch_loss / len(train_dataloader.dataset)
        print("Epoch {}/{}\n\tTrain accuracy = {:.3f}\n\tAverage loss = {:.3f}".format(epoch, max_epochs, train_accuracy,
                                                                                     average_loss))

        # nacrtati filtar u prvom sloju
        draw_conv_filters(epoch, model.conv1, save_dir, f'lambda={model.weight_decay}')

        validation_accuracy, validation_loss = evaluate("Validation", valid_dataloader, model, config)

        train_accuracy_array.append(train_accuracy)
        train_loss_array.append(average_loss)
        validation_accuracy_array.append(validation_accuracy)
        validation_loss_array.append(validation_loss)

    return model, train_accuracy_array, validation_accuracy_array, train_loss_array, validation_loss_array


def evaluate(name, dataLoader, model, config):
    print("\nEvaluacija za:", name)
    save_dir = config['save_dir']
    with torch.no_grad():
        # postavi model u evaluacijski mode
        model.eval()

        y_true_array = []
        y_pred_array = []

        total_loss = 0
        cnt_correct = 0

        for (x, y_true) in dataLoader:
            outputs = model.forward(x)
            _, y_pred = outputs.max(1)

            y_true_array.extend(y_true)
            y_pred_array.extend(y_pred)

            cnt_correct = cnt_correct + prebrojiTocne(y_pred, y_true)
            loss_i = nn.CrossEntropyLoss()(outputs, y_true)
            total_loss = total_loss + loss_i

        average_loss = total_loss / len(dataLoader.dataset)
        average_correct = cnt_correct / len(dataLoader.dataset)

        print("Prosječni gubitak = {:.3f}\nTočnost = {:.3f}\n".format(average_loss, average_correct))

        return average_correct, average_loss


if __name__ == "__main__":

    DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
    SAVE_DIR = Path(__file__).parent / 'out' / 'treci'

    filename = SAVE_DIR / 'output.txt'

    config = {}
    config['max_epochs'] = 50
    config['batch_size'] = 50
    config['save_dir'] = SAVE_DIR
    config['weight_decay'] = 1e-3
    config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}
    config['learning_rate'] = 1e-4

    # hiperparametri
    weight_decays = [0, 1e-3, 1e-2, 1e-1]

    dataset_train, dataset_test = MNIST(DATA_DIR, train=True, download=True, transform=ToTensor()), MNIST(DATA_DIR, train=False,
                                                                                                transform=ToTensor())
    dataset_train, dataset_validate = random_split(dataset_train, [0.8, 0.2], generator=torch.Generator().manual_seed(10))

    # initialize the train, validation, and test data loaders
    train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=config['batch_size'])
    valid_dataloader = DataLoader(dataset_validate, batch_size=config['batch_size'])
    test_dataloader = DataLoader(dataset_test, batch_size=config['batch_size'])


    for i, weight_decay in enumerate(weight_decays):
        print("Treniranje mreze s lambda = {}".format(weight_decay))
        config['weight_decay'] = weight_decay

        # definiraj model
        model = Net(weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # treniranje modela
        model, train_accuracy_array, validation_accuracy_array, train_loss_array, validation_loss_array = train(train_dataloader, valid_dataloader, model, config)

        # evaluacija modela
        evaluation_accuracy, evaluation_loss = evaluate("Ispitivanje", test_dataloader, model, config)

        # ispis u datoteku
        if i == 0:
            f = open(filename, "w")
        else:
            f = open(filename, "a")

        output_string = f'lambda = {weight_decay}\n'
        f.write(output_string)
        output_string = ""
        for e, (t_a, v_a) in enumerate(zip(train_accuracy_array, validation_accuracy_array)):
            output_string += f'\tepoch {e+1}\n\t\ttrain accuracy = {t_a}\n\t\tvalidation accuracy = {v_a}\n'
        f.write(output_string)
        output_string = f'evaluation accuracy = {evaluation_accuracy}\n\n'
        f.write(output_string)
        f.close()

        # crtanje grafa epohe-loss
        nacrtaj_i_spremi_graf(np.arange(1, config['max_epochs']+1, 1), train_loss_array, config['save_dir'], f'weight decay = {weight_decay}', "epoch", "loss")
