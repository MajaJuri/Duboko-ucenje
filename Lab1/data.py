import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


class Random2DGaussian:
    minx = 0
    maxx = 10
    miny = 0
    maxy = 10
    scalecov = 5

    def __init__(self):
        # slucajno odabrati sredinu razdiobe
        mean = np.random.random_sample(2)*(self.maxx-self.minx, self.maxy-self.miny)
        mean += (self.minx, self.miny)
        # slucajno odabrati svojstvene vrijednosti kovarijacijske matrice
        eigenvals = ((np.random.random_sample() * (self.maxx - self.minx) / self.scalecov) ** 2,
                     (np.random.random_sample() * (self.maxy - self.miny) / self.scalecov) ** 2)
        # slucajno odabrati kut rotacije kovarijacijske matrice
        kut = np.random.random_sample()*2*np.pi
        r = [[np.cos(kut), -np.sin(kut)], [np.sin(kut), np.cos(kut)]]
        # matricu sigma dobijemo kao umnozak transponirane matrice R, matrice D i matrice R
        sigma_matrica = np.dot(np.dot(np.transpose(r), np.diag(eigenvals)), r)
        # funkcija multivariate_normal sluzi za uzorkovanje
        self.get_sample = lambda n: np.random.multivariate_normal(mean, sigma_matrica, n)

def sample_gauss_2d(nclasses, nsamples):
    # create the distributions and groundtruth labels
    Gs = []
    Ys = []
    for i in range(nclasses):
        Gs.append(Random2DGaussian())
        Ys.append(i)

    # sample the dataset
    X = np.vstack([G.get_sample(nsamples) for G in Gs])
    Y_ = np.hstack([[Y] * nsamples for Y in Ys])

    return X, Y_

def eval_AP(labels):
    """Recovers AP from ranked labels"""
    ranked_labels = np.argsort(labels)
    n = len(ranked_labels)
    pos = sum(ranked_labels)
    neg = n - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec = 0
    # IPython.embed()
    for x in ranked_labels:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if x:
            sumprec += precision

        # print (x, tp,tn,fp,fn, precision, recall, sumprec)
        # IPython.embed()

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec / pos

def eval_perf_binary(Y, Y_):
  tp = sum(np.logical_and(Y==Y_, Y_==True))
  fn = sum(np.logical_and(Y!=Y_, Y_==True))
  tn = sum(np.logical_and(Y==Y_, Y_==False))
  fp = sum(np.logical_and(Y!=Y_, Y_==False))
  recall = tp / (tp + fn)
  precision = tp / (tp + fp)
  accuracy = (tp + tn) / (tp+fn + tn+fp)
  return accuracy, recall, precision

def class_to_onehot(Y):
  Yoh=np.zeros((len(Y),max(Y)+1))
  Yoh[range(len(Y)),Y] = 1
  return Yoh

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

# function - surface to be plotted
# rect - domena funckije [x_min, y_min], [x_max, y_max]
# offset:   the level plotted as a contour plot
def graph_surface(function, rect, offset=0.5, width=256, height=256):
    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = np.array(function(grid)).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])

def my_graph_data(X, Y_, Y):
    colors= ["red",    "green",    "blue",    "cyan",    "magenta",    "yellow",    "black",    "white",    "gray",    "lightgray",    "darkgray",    "orange",    "purple",    "pink",    "brown"]
    #print(Y_)
    #print(Y)
    for index in range(len(X)):
        x = X[index]
        y_true = Y_[index]
        y_pred = Y[index]
        #print(x)
        #print(y_true)
        #print(y_pred)

        if y_true == y_pred:
            m = 'o'
        else:
            m = 's'
        #print(m)

        c=colors[y_true]
        #print(c)
        #print()

        plt.scatter(x[0], x[1], color=c, marker=m, edgecolors='black')

# stvara scatter plot
# X - data
# Y_ - y_true
# Y - y_pred
def graph_data(X, Y_, Y, special=[]):
    # colors of the datapoint markers
    palette = ([0.5, 0.5, 0.5], [1, 1, 1], [0.2, 0.2, 0.2])
    colors = np.tile([0.0, 0.0, 0.0], (Y_.shape[0], 1))
    for i in range(len(palette)):
        colors[Y_ == i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_ == Y)
    plt.scatter(X[good, 0], X[good, 1], c=colors[good],
                s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_ != Y)
    plt.scatter(X[bad, 0], X[bad, 1], c=colors[bad],
                s=sizes[bad], marker='s', edgecolors='black')

# K - broj komponenti
# C - broj klasa
# N - broj uzoraka
# ncomponents, nclasses, nsamples
def sample_gmm_2d(K, C, N):
    # create the distributions and groundtruth labels
    Gs = []
    Ys = []
    for i in range(K):
        Gs.append(Random2DGaussian())
        Ys.append(np.random.randint(C))

    # sample the dataset
    X = np.vstack([G.get_sample(N) for G in Gs])
    Y_ = np.hstack([[Y] * N for Y in Ys])
    return X, Y_



def myDummyDecision(X):
  scores = X[:,0] + X[:,1] - 5
  return scores



def my_eval_perf_multi(Y_true,Y_pred):
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


def gimme_one_hot(Y):
    Yoh = []
    for y in Y:
        klasa = np.argmax(y)
        red = []
        for index in range(np.max(Y)):
            if index == klasa:
                red.append(1)
            else:
                red.append(0)
        Yoh.append(red)
    return Yoh

def print_performance_indicators(y_pred, y_true):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    response = metrics.recall_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    avg_precision = metrics.average_precision_score(y_true, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    print("Accuracy: {:.3f}".format(accuracy))
    print("Response: {:.3f}".format(response))
    print("Precision: {:.3f}".format(precision))
    print("Average Precision: {:.3f}".format(avg_precision))
    print("Confusion matrix:\n {}".format(confusion_matrix))

if __name__=="__main__":
    np.random.seed(100)

    X, Y_ = sample_gmm_2d(K=4, C=2, N=30)
    Y = myDummyDecision(X)>0.5
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(myDummyDecision, rect, offset=0)
    my_graph_data(X, Y_, Y)
    plt.show()