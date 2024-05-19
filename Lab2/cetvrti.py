import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
import math
import skimage as ski
import os
from pathlib import Path

from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10


def nacrtaj_graf(loss_train, loss_validation, accuracy_train, accuracy_validation, learning_rates, config):
    save_dir = config['save_dir']
    broj_epoha = len(loss_train)
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    axs[0].set_title("Average class accuracy")
    axs[0].plot(np.arange(1, broj_epoha+1, 1), accuracy_train, color='m', label="train", marker='o')
    axs[0].plot(np.arange(1, broj_epoha+1, 1), accuracy_validation, color='c', label="validation", marker='o')
    axs[1].set_title("Cross-entropy loss")
    axs[1].plot(np.arange(1, broj_epoha+1, 1), loss_train, color='m', label="train", marker='o')
    axs[1].plot(np.arange(1, broj_epoha+1, 1), loss_validation, color='c', label="validation", marker='o')
    axs[2].set_title("Learning rate")
    axs[2].plot(np.arange(1, broj_epoha+1, 1), learning_rates, color='m', label="learning rate", marker='o')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    fig_title = save_dir / 'graf.png'
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(fig_title)


def draw_image(img, mean, std, save_dir, filename):
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, filename))


def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]


def draw_conv_filters(epoch, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d.png' % (epoch)
  ski.io.imsave(os.path.join(save_dir, filename), img)


def nacrtaj_sliku(img, mean, std):
    img = img.transpose(1, 2, 0)
    img = img * std
    img = img + mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()


def prebrojiTocne(y_pred, y_true):
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    cnt_correct = np.sum(y_pred_np == y_true_np)
    return cnt_correct


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2, bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        h = self.conv1(x.float())
        h = self.relu1(h)
        h = self.pool1(h)

        h = self.conv2(h)
        h = self.relu2(h)
        h = self.pool2(h)

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = self.relu3(h)

        h = self.fc2(h)
        h = self.relu4(h)

        y = self.fc3(h)
        return y


def train(train_dataloader, validate_dataloader, model, config):
    print("\nTreniranje...")
    save_dir = config['save_dir']
    l2_lambda = config['weight_decay']
    max_epochs = config['max_epochs']

    optimizer = torch.optim.Adam(model.parameters(), lr=l2_lambda)  # adam ili sgd
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    learning_rates = []
    train_loss_array = []
    train_accuracy_array = []
    validation_accuracy_array = []
    validation_loss_array = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_epoch_correct = 0
        total_epoch_loss = 0

        for i, (x, y_true) in enumerate(train_dataloader):
            #print("y_true", y_true)
            #print("y_true.shape", y_true.shape)
            outputs = model.forward(x)
            #print("outputs", outputs)
            #print("outputs.shape", outputs.shape)

            loss = nn.CrossEntropyLoss()(outputs, y_true)

            _, y_pred = outputs.max(1) # vraca max vrijednost u svakom redu
            #print("y_pred", y_pred)

            # loss + opzimize
            optimizer.zero_grad()
            loss.backward()
            total_epoch_loss = total_epoch_loss + loss.item()
            optimizer.step()

            # izracunaj broj tocnih
            total_epoch_correct = total_epoch_correct + prebrojiTocne(y_pred, y_true)
            #print(total_epoch_correct)

            # za svaki stoti batch ispisati loss
            if i % 100 == 0 and i > 0:
                print('epoch %d, step %d, item loss = %.2f' %
                      (epoch, i, loss.item()))

        # na kraju svake epohe nacrtati filtar
        draw_conv_filters(epoch, model.conv1.weight.detach().numpy(), save_dir)

        # izracunati accuracy i tocnost
        train_accuracy = (total_epoch_correct / len(train_dataloader.dataset))
        average_loss = total_epoch_loss / len(train_dataloader.dataset)
        print("Epoch {}/{}\n\tTrain accuracy = {:.3f}\n\tAverage loss = {:.3f}".format(epoch, max_epochs, train_accuracy, average_loss))

        # promijeniti learning_rate
        lr_scheduler.step()
        learning_rates.append(lr_scheduler.get_last_lr()[0])

        # nakon svake epohe ucenja pratiti napredak pomocu funkcije evaluate na skupu za validaciju
        validation_accuracy, validation_loss = evaluate("Validacija", model, validate_dataloader, config)

        train_accuracy_array.append(train_accuracy)
        train_loss_array.append(average_loss)
        validation_accuracy_array.append(validation_accuracy)
        validation_loss_array.append(validation_loss)

    return model, train_accuracy_array, validation_accuracy_array, validation_loss_array, train_loss_array, learning_rates


def evaluate(name, model, dataLoader, config):
    print("\nEvaluacija za:", name)
    save_dir = config['save_dir']
    save_dir = save_dir / 'netocno_klasificirane_slike_20'
    krivo_klasificirane_slike = []
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

            for i in range(y_true.size(0)):
                if y_pred[i] != y_true[i]:
                    krivo_klasificirane_slike.append((x[i], y_pred[i], y_true[i], loss_i))

        matrica_konfuzije = confusion_matrix(y_true_array, y_pred_array)
        print("Matrica konfuzije:\n", matrica_konfuzije)
        for i in range(10):
            TP = matrica_konfuzije[i, i]
            FP = np.sum(matrica_konfuzije[:, i]) - TP
            FN = np.sum(matrica_konfuzije[i, :]) - TP
            TN = np.sum(matrica_konfuzije) - TP - FN - FP
            print("Klasa", i+1)
            print("\tPreciznost = {:.3f}".format(TP/(TP+FP)))
            print("\tOdziv = {:.3f}".format(TP/(TP+FN)))

        average_loss = total_loss / len(dataLoader.dataset)
        average_correct = cnt_correct / len(dataLoader.dataset)

        print("Prosječni gubitak = {:.3f}\nTočnost = {:.3f}\n".format(average_loss, average_correct))

        if name == "Ispitivanje":
            krivo_klasificirane_slike.sort(key=lambda x: x[3], reverse=True)
            for i in range(20):
                image, predicted, true_label, loss = krivo_klasificirane_slike[i]
                predicted_prob = torch.softmax(outputs[i], dim=0)
                top3_classes = predicted_prob.argsort(dim=0, descending=True)[:3]

                #plt.title(f"Slika {i + 1}")
                #plt.imshow(image.permute(1, 2, 0))
                #plt.show()

                print(f"Slika {i + 1}:")
                print("y_pred: {}, top 3 klase: {}, y_true: {}, loss: {:.3F}".format(predicted, top3_classes, true_label, loss.item()))
                draw_image(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], save_dir, f"Slika_{i+1}.png")

        return average_correct, average_loss


if __name__ == "__main__":

    DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
    SAVE_DIR = Path(__file__).parent / 'out' / 'cetvrti'

    filename = SAVE_DIR / 'output.txt'

    config = {}
    config['max_epochs'] = 50
    config['batch_size'] = 50
    config['save_dir'] = SAVE_DIR
    config['weight_decay'] = 1e-3
    config['lr_policy'] = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}
    config['learning_rate'] = 1e-1
    # hiperparametri
    batch_size = config['batch_size']
    num_epoch = 8
    weight_decay = 0.001

    img_height = 32
    img_width = 32

    # ucitavanje podataka
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = CIFAR10(root='./data', train=True, download=True, transform=transforms)

    dataset_test = CIFAR10(root='./data', train=False, download=True, transform=transforms)

    dataset_train, dataset_valid = random_split(dataset_train, [0.80, 0.20], generator=torch.Generator().manual_seed(10))

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size)

    # stvaranje modela
    model = CIFAR10Model()

    # ucenje modela
    model, train_accuracy_array, validation_accuracy_array, validation_loss_array, train_loss_array, learning_rates = \
                                                                                    train(train_dataloader, valid_dataloader, model, config)

    # ispitivanje modela
    evaluation_accuracy, evaluation_loss = evaluate("Ispitivanje", model, test_dataloader, config)

    # ispis u datoteku
    f = open(filename, "a")
    output_string = ""
    for e, (t_a, v_a, v_l, lr) in enumerate(zip(train_accuracy_array, validation_accuracy_array, validation_loss_array, learning_rates)):
        output_string += f'epoch {e + 1}\n\ttrain accuracy = {t_a}\n\tvalidation accuracy = {v_a}\n\tlearning rate = {lr}\n'
    f.write(output_string)
    output_string = f'evaluation accuracy = {evaluation_accuracy}\n\n'
    f.write(output_string)
    f.close()

    # crtanje grafa
    nacrtaj_graf(train_loss_array, validation_loss_array, train_accuracy_array, validation_accuracy_array, learning_rates, config)




