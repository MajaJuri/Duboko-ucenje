import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import data


class PTDeep(nn.Module): # nasljedivanje osnovnog razreda torch.nn.Module
    def __init__(self, slojevi, aktivacijska_funkcija=torch.relu, param_lambda=0): # konstruktor
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()
        self.layers = slojevi
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.param_lambda = param_lambda
        for index in range(len(self.layers)-1):
            self.weights.append(nn.Parameter(torch.randn(self.layers[index], self.layers[index+1])))
            self.biases.append(nn.Parameter(torch.randn(1, self.layers[index+1])))
        self.f = aktivacijska_funkcija
       #for weight in self.weights:
        #    print(weight.data)
        #print()
        #for bias in self.biases:
        #    print(bias.data)

    def forward(self, X): # mora se definirati metoda forward
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        # koristiti: torch.mm, torch.softmax

        Z = torch.mm(X.double(), self.weights[0].double()) + self.biases[0].double() # torch.mm -> množenje matrica
        h = self.f(Z)
        for hidden_index in range(1, len(self.layers)-1):
            Z = torch.mm(h.double(), self.weights[hidden_index].double()) + self.biases[hidden_index].double()
            h = self.f(Z)

        probs = torch.softmax(h, dim=1)
        return probs

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        # koristiti: torch.log, torch.mean, torch.sum
        probs = self.forward(X)

        # cross-entropy gubitak
        # Y_oh * torch.log(Y) -> daje log izglednost te klase koja je zapravo točna
        # pozbrajamo po redovima da dobijemo tu vrijednost za svaki input
        # na kraju izracunamo srednju vrijednost tih gubitaka
        #print(Yoh_)
        #print(probs)
        loss_cross_entropy = -torch.mean(torch.sum(Yoh_ * torch.log(probs), dim=1))

        # calculate L2 regularization
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.sum(param.pow(2))

        # Dodajte regularizaciju na način da gubitak formulirate kao zbroj unakrsne entropije i L2 norme vektorizirane matrice težina pomnožene hiperparametrom param_lambda.
        total_loss = loss_cross_entropy + 0.5 * self.param_lambda * l2_reg

        return total_loss

    def count_params(self):
        print("Konfiguracija: {}".format(self.layers))
        total_params = 0
        for (name, param) in self.named_parameters():
            print("Dimension of {}: {}".format(name, tuple(param.shape)))
            total_params += param.numel()
        print("Total number of parameters: {}\n".format(total_params))


def train(model, X, Yoh_, param_niter, param_delta):
    optimizer = optim.SGD(model.parameters(), lr=param_delta)
    losses = []
    for i in range(param_niter):
        # računanje gubitka i gradijenta
        loss = model.get_loss(X, Yoh_) / len(X)
        loss.backward()

        # ažuriranje parametara
        optimizer.step()
        # resetiranje gradijenta
        optimizer.zero_grad()
        # ispis gubitka
        if i%10 == 0:
            print(f'step: {i}, loss:{loss.item()}')
        losses.append(loss.item())
    return_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            return_params.append(param)
    return return_params, losses


def logreg_classify(probs):
    Y_pred = []
    for prob in probs:
        Y_pred.append(np.argmax(prob))
    return Y_pred


def ptdeep_decfun(model):
    # X -> array
    def classify(X):
        probs = eval(model, X)
        Y_pred = []
        for prob in probs:
            Y_pred.append(np.argmax(prob))
        return Y_pred
    return classify


def eval(model, X):
    # X -> array
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    #X = X.clone().detach().requires_grad_(True).float()
    probs = model.forward(torch.from_numpy(X))
    return probs.detach().numpy()


def iscrtaj_granicu(X, model):
    # Create a mesh to plot in
    r = 0.04  # mesh resolution
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, r), np.arange(y_min, y_max, r))
    XX = np.c_[xx.ravel(), yy.ravel()]
    XX_torch = torch.from_numpy(XX)
    Z = []
    probs = eval(model, XX_torch)
    for prob in probs:
        Z.append(np.argmax(prob))
    #print(Z)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    param_niter = 10000
    param_delta = 0.1
    param_lambda = 1e-4

    #Izvedite razred PTDeep te isprobajte konfiguraciju [2, 3] na istim podatcima kao i u prethodnom zadatku
    K = 5
    C = 3 # broj klasa
    N = 20
    X, Y_true = data.sample_gmm_2d(K=K, C=C, N=N)
    Yoh_ = data.class_to_onehot(Y_true)
    Yoh_ = torch.from_numpy(Yoh_)
    ptdeep = PTDeep([2, 3], torch.relu, param_lambda=param_lambda)
    named_parameters, _ = train(ptdeep, torch.from_numpy(X), Yoh_, param_niter, param_delta)
    ptdeep.count_params()
    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptdeep, X)
    Y_pred = logreg_classify(probs)
    # ispiši performansu (preciznost i odziv po razredima)
    data.my_eval_perf_multi(Y_true, Y_pred)

    #K = [4, 6] # dimenzija podataka
    #C = [2, 2] # broj klasa
    #N = [40, 10]
    #konfiguracije = [[2, 10, 10, 2], [2, 2], [2, 10, 2]]

    K = [4]
    C = [2] # broj klasa
    N = [40]
    konfiguracije = [[2, 10, 10, 2]]

    for index in range(len(K)):
        X, Y_true = data.sample_gmm_2d(K=K[index], C=C[index], N=N[index])
        rect = (np.min(X, axis=0), np.max(X, axis=0))
        Yoh_ = data.class_to_onehot(Y_true)
        # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
        Yoh_ = torch.from_numpy(Yoh_)
        for konfiguracija in konfiguracije:
            # definiraj model:
            ptdeep = PTDeep(konfiguracija, torch.relu)
            named_parameters, _ = train(ptdeep, torch.from_numpy(X), Yoh_, param_niter, param_delta)

            # dohvati vjerojatnosti na skupu za učenje
            probs = eval(ptdeep, X)
            Y_pred = logreg_classify(probs)

            ptdeep.count_params()
            # ispiši performansu (preciznost i odziv po razredima)
            data.print_performance_indicators(Y_pred, Y_true)

            # iscrtaj rezultate, decizijsku plohu
            #iscrtaj_granicu(X, ptdeep)
            data.graph_surface(ptdeep_decfun(ptdeep), rect)
            data.graph_data(X, Y_true, Y_pred)
            plt.title("K = {}, C = {}, N = {}, konfiguracija = {}".format(K[index], C[index], N[index], konfiguracija))
            plt.show()

