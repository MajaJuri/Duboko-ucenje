import torch
import torch.nn as nn
import time
import torch.optim
from Zadatak1 import MNISTMetricDataset
from torch.utils.data import DataLoader
from Zadatak2 import SimpleMetricEmbedding
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        feats = img.view(img.size(0), -1)
        return feats


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 28*28
    model = IdentityModel().to(device)

    t0 = time.time_ns()
    if EVAL_ON_TEST or EVAL_ON_TRAIN:
        print("Computing mean representations for evaluation...")
        representations = compute_representations(model, train_loader, num_classes, emb_size, device)
    if EVAL_ON_TRAIN:
        print("Evaluating on training set...")
        acc1 = evaluate(model, representations, traineval_loader, device)
        print(f"Train Top1 Acc: {round(acc1 * 100, 2)}%")
    if EVAL_ON_TEST:
        print("Evaluating on test set...")
        acc1 = evaluate(model, representations, test_loader, device)
        print(f"Test Accuracy: {acc1 * 100:.2f}%")
    t1 = time.time_ns()
    print(f"Time (sec): {(t1 - t0) / 10 ** 9:.1f}")