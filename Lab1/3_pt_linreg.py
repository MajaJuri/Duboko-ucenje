import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def iscrtaj_lin_granicu(X, a, b):
    x = np.linspace(np.min(X)-1, np.max(X)+1, 100)
    y = a * x + b

    plt.plot(x, y) # plot the line
    plt.title('y = {}x + {}'.format(a, b))


def iscrtaj_tocke(X, Y_true):
    plt.scatter(X, Y_true)


## Definicija računskog grafa
# podaci i parametri, inicijalizacija parametara
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

X = []
Y = []

#unos tocaka
while True:
    x_str = input("Upisi vrijednost x: (ili 'done' ako si gotov s upisivanjem) ")
    if x_str == 'done':
        break

    y_str = input("Upisi tocnu Y vrijednost tocke: ")
    try:
        x = float(x_str)
        y = float(y_str)
    except ValueError:
        print("Upisi broj")
        continue

    X.append(x)
    Y.append(y)

X = torch.tensor(X)
Y = torch.tensor(Y)

# optimizacijski postupak: gradijentni spust
# u SGD dajemo parametre koje optimiziramo
optimizer = optim.SGD([a, b], lr=0.001)
loss = np.inf

for i in range(100):
#while loss > 1e-5:
    # afin regresijski model
    Y_ = a*X + b # Y_pred

    diff = (Y-Y_)

    # kvadratni gubitak
    loss = torch.sum(diff ** 2) / len(X)  # podijeli s brojem podataka kako ne bi ovisilo o broju podataka

    # računanje gradijenata
    loss.backward()
    # least squared error = 1/N * (Y_pred - Y_true)^2 = 1/N * diff^2
    # derivacija po a -> parcijalna derivacija po diff ili Y_pred pa Y_pred po
    # isto za b
    dL_da = -torch.sum(2*diff*X)/len(X)
    dL_db = -torch.sum(2*diff*1)/len(X)

    # ispis gradijenta za a i b
    print("step: {}, gradient_a: {}, gradient_b: {}".format(i, a.grad.data, b.grad.data))
    print("\tANALITICKI IZRACUN => gradient_a: {:.4f}, gradient_b: {:.4f}".format(dL_da, dL_db))

    # korak optimizacije
    optimizer.step()

    # Postavljanje gradijenata na nulu
    optimizer.zero_grad()

    print("\tloss:{}, Y_:{}, a:{}, b {}".format(loss, Y_, a, b))
    print()

X = X.detach().numpy()
Y_ = Y_.detach().numpy()
Y = Y.detach().numpy()
iscrtaj_lin_granicu(X, a.detach().numpy(), b.detach().numpy())
iscrtaj_tocke(X, Y)
plt.xlabel('x')  # add x-axis label
plt.ylabel('y')  # add y-axis label
plt.grid()
plt.show()
