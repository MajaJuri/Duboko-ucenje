import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(num_maps_in),
            nn.ReLU(),
            nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k),
        )


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        self.bnrc1 = _BNReluConv(input_channels, emb_size, k=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bnrc2 = _BNReluConv(emb_size, emb_size, k=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bnrc3 = _BNReluConv(emb_size, emb_size, k=3)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=2)
        self.margin = 1.0

    def get_features(self, img):
        x = self.bnrc1(img)  # [BATCH_SIZE=64, EMB_SIZE=32, 26, 26]
        x = self.max_pool1(x)  # [BATCH_SIZE=64, EMB_SIZE=32, 12, 12]
        x = self.bnrc2(x)  # [BATCH_SIZE=64, EMB_SIZE=32, 10, 10]
        x = self.max_pool2(x)  # [BATCH_SIZE=64, EMB_SIZE=32, 4, 4]
        x = self.bnrc3(x)  # [BATCH_SIZE=64, EMB_SIZE=32, 2, 2]
        x = self.avg_pool3(x)  # [BATCH_SIZE=64, EMB_SIZE=32, 1, 1]
        x = x.view(x.shape[0], -1)  # [BATCH_SIZE=64, EMB_SIZE=32]
        print("end")
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        dap = torch.norm(a_x - p_x, dim=1)
        dan = torch.norm(a_x - n_x, dim=1)
        loss = torch.relu(dap - dan + self.margin).mean()
        #loss = F.triplet_margin_loss(a_x, p_x, n_x,)
        return loss

