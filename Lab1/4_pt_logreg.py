import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt

class PTLogreg(nn.Module): # nasljedivanje osnovnog razreda torch.nn.Module
    def __init__(self, D, C, param_lambda=0): # konstruktor
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()
        print(param_lambda)
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.randn(1, C))
        self.param_lambda = param_lambda

    def forward(self, X): # mora se definirati metoda forward
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        # koristiti: torch.mm, torch.softmax
        Z = torch.mm(X.double(), self.W.double()) + self.b.double() # torch.mm -> množenje matrica
        Y = torch.softmax(Z, dim=1)
        return Y

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        # koristiti: torch.log, torch.mean, torch.sum
        Y = self.forward(X)

        # cross-entropy gubitak
        # Y_oh * torch.log(Y) -> daje log izglednost te klase koja je zapravo točna
        # pozbrajamo po redovima da dobijemo tu vrijednost za svaki input
        # na kraju izracunamo srednju vrijednost tih gubitaka
        loss_cross_entropy = -torch.mean(torch.sum(Yoh_ * torch.log(Y), dim=1))

        # calculate L2 regularization
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.sum(param.pow(2))

        # Dodajte regularizaciju na način da gubitak formulirate kao zbroj unakrsne entropije i L2 norme vektorizirane matrice težina pomnožene hiperparametrom param_lambda.
        total_loss = loss_cross_entropy + self.param_lambda * l2_reg

        return total_loss

    def train(self, X, Yoh_, param_niter, param_delta):
        """Arguments:
           - X: model inputs [NxD], type: torch.Tensor
           - Yoh_: ground truth [NxC], type: torch.Tensor
           - param_niter: number of training iterations
           - param_delta: learning rate
        """
        optimizer = optim.SGD(self.parameters(), lr=param_delta)
        for i in range(param_niter):
            # računanje gubitka i gradijenta
            loss = self.get_loss(X, Yoh_) / len(X)
            loss.backward()

           # ispis gradijenta za svaki parametar u svakoj iteraciji
           # for name, param in self.named_parameters():
           #     if param.requires_grad:
           #         print(name, param.grad)

            # ažuriranje parametara
            optimizer.step()
            # resetiranje gradijenta
            optimizer.zero_grad()
            # ispis gubitka
        print(f'step: {i}, loss:{loss.item()}')
        return_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                return_params.append(param)
        print(return_params)
        return return_params

    def eval(self, X):
        #print(X)
        """Arguments:
           - model: type: PTLogreg
           - X: actual datapoints [NxD], type: np.array
           Returns: predicted class probabilites [NxC], type: np.array
        """
        #X = X.clone().detach().requires_grad_(True).float()
        Y_pred = self.forward(torch.from_numpy(X))
        return Y_pred.detach().numpy()

def eval_perf_multi(Y_true,Y_pred):
    if len(Y_true) != len(Y_pred):
        print("Y_true i Y_pred trebaju biti iste duljine")
        return
    else:
        confusion_matrices = []
        broj_klasa = len(np.unique(np.append(Y_true, Y_pred)))
        #print(broj_klasa)
        for klasa in range(broj_klasa):
            confusion_matrix = []
            TP = TN = FP = FN = 0
            for index in range(len(Y_pred)):
                if Y_pred[index] == klasa:
                    if Y_true[index] == klasa:
                        TP+=1
                    else:
                        if Y_true[index] != klasa:
                            FP+=1
                else:
                    if Y_true[index] == klasa:
                        FN += 1
                    else:
                        TN += 1
            print("Klasa {}".format(klasa))
            red = []
            red.append(TP)
            red.append(FP)
            print("\t{}".format(red))
            confusion_matrix.append(red)
            red = []
            red.append(FN)
            red.append(TN)
            print("\t{}".format(red))
            confusion_matrix.append(red)
            confusion_matrices.append(confusion_matrix)
            if TP+FP != 0:
                preciznost = TP/(TP+FP)
                print("\n\tPreciznost = {:.2f}".format(preciznost))
            if TP+FN != 0:
                odziv = TP/(TP+FN)
                print("\tOdziv = {:.2f}".format(odziv))
            print()

def logreg_classify(probs):
    Y_pred = []
    for prob in probs:
        Y_pred.append(np.argmax(prob))
    return Y_pred

def logreg_decfun(model):
    def classify(X):
        probs = model.eval(X)
        return logreg_classify(probs)
    return classify

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    K = 5
    C = 3 # broj klasa
    N = 20
    X, Y_true = data.sample_gmm_2d(K=K, C=C, N=N)
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    Yoh_ = data.class_to_onehot(Y_true)

    param_niter = 1000
    param_delta = 0.01
    param_lambda = [0, 0.01, 0.1]

    # definiraj model:
    for lambd in param_lambda:
        ptlr = PTLogreg(X.shape[1], Yoh_.shape[1], lambd)

        # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
        X_torch = torch.from_numpy(X)
        Yoh_torch = torch.from_numpy(Yoh_)
        named_parameters = ptlr.train(X_torch, Yoh_torch, param_niter, param_delta)

        # dohvati vjerojatnosti na skupu za učenje
        # ovdje X mora biti np.array
        probs = ptlr.eval(X)
        Y_pred = logreg_classify(probs)

        # ispiši performansu (preciznost i odziv po razredima)
        eval_perf_multi(Y_true, Y_pred)

        # iscrtaj rezultate, decizijsku plohu
        data.graph_surface(logreg_decfun(ptlr), rect)
        data.graph_data(X, Y_true, Y_pred)
        plt.title("lambda={}".format(lambd))
        plt.show()

    # nece imati rjesenje ako stavimo da je learning rate prevelik

