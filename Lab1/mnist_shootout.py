import matplotlib.pyplot as plt
import torch
import torchvision
import ptdeep_5 as ptdeep
import data
import numpy as np

dataset_root = '/tmp/mnist'  # change this to your preference
mnist_dataset = torchvision.datasets.MNIST(dataset_root, download=True)
mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

x_train, y_train = mnist_train.data, mnist_train.targets
x_test, y_test = mnist_test.data, mnist_test.targets
x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

img, label = mnist_dataset[0]

print(img)
print(label)

SUBPLOT_ROWS = 3
SUBPLOT_COLS = 3

fig, ax = plt.subplots(SUBPLOT_ROWS, SUBPLOT_COLS)
fig.tight_layout()

for i in range(SUBPLOT_ROWS):
  for j in range(SUBPLOT_COLS):
    img, label = mnist_dataset[i * SUBPLOT_COLS + j]
    ax[i, j].imshow(img, cmap='gray_r')
    ax[i, j].set_title(f'LABEL : {label}')

plt.show()

VAL_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.2
TRAIN_PERCENTAGE = 1.0 - VAL_PERCENTAGE - TEST_PERCENTAGE

# podijeli u train i test
val_size = int(VAL_PERCENTAGE * len(mnist_dataset))
test_size = int(TEST_PERCENTAGE * len(mnist_dataset))
train_size = len(mnist_dataset) - val_size - test_size

N = x_train.shape[0]
D = x_train.shape[1] * x_train.shape[2]
C = y_train.max().add_(1).item()
print(f'Broj podataka = {N}')
print(f'Broj klasa = {C}')
print(f'D = {D}')

mnist_train_dataset, mnist_val_dataset, mnist_test_dataset = torch.utils.data.random_split(mnist_dataset, [train_size, val_size, test_size])
print(f'Prije podjele: {len(mnist_dataset)}')
print(f'Nakon podjele:  {len(mnist_train_dataset)}(train) + {len(mnist_val_dataset)}(val) + {len(mnist_test_dataset)}(test)')


## 1. tocka
# create a PTDeep model with configuration [784, 10]
lambde = [0, 0.1, 0.01]
iters = [10, 100, 1000]
konfiguracije = [[784, 10], [784, 100, 10]]
param_delta = 0.1


for iter in iters:
    for lambd in lambde:
        for konf in konfiguracije:
            print("Konfiguracija={} broj iteracija={} lambda={}".format(konf, iter, lambd))
            model = ptdeep.PTDeep(konf, param_lambda=lambd)
            Yoh = data.class_to_onehot(y_train)
            Yoh = torch.from_numpy(Yoh)
            params, losses = ptdeep.train(model, x_train.view(-1, 784), Yoh, iter, param_delta)
            #print(losses)
            model.eval()  # put model in evaluation mode
            probs = model.forward(x_train.view(-1, 784))
            Y_pred = np.argmax(probs.detach().numpy(), axis=1)
            print("Y_true ", y_train.detach().numpy())
            print("Y_pred ", Y_pred)
            Yoh_test = data.class_to_onehot(Y_pred)
            with torch.no_grad():  # turn off gradient calculation
                for i in range(10):  # iterate over each digit
                    digit_indices = torch.where(y_test == i)[0]  # get indices of test examples with the current digit label
                    digit_examples = x_test[digit_indices]  # get the examples with the current digit label
                    digit_weights = model.weights[-1][:, i]  # get the weights for the current digit label
                    #print(f"Weight matrix for digit {i}:")
                    #print(digit_weights.reshape(28, 28))  # MNIST slike su 28x28
            #print(losses)
            plt.plot(np.log(losses)) #, label="Konfiguracija={} broj iteracija={} lambda={}".format(konf, iter, lambd)
            plt.title("Konfiguracija={} broj iteracija={} lambda={}".format(konf, iter, lambd))
            plt.show()


