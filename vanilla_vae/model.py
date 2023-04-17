import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def imsave(name, img):
    npimg = img.numpy()
    plt.imsave(name, np.transpose(npimg, (1, 2, 0)))


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.eConv1 = nn.Conv2d(3, 6, 5, stride=2)
        self.eConv2 = nn.Conv2d(6, 12, 7, dilation=1)
        self.ePool1 = nn.MaxPool2d(2, 2)
        self.eConv3 = nn.Conv2d(12, 24, 5)
        self.ePool2 = nn.MaxPool2d(2, 2)
        self.eF1 = nn.Linear(24 * 28 * 28, 512)
        self.eMu = nn.Linear(512, 512)
        self.eSigma = nn.Linear(512, 512)

        self.dConvT1 = nn.ConvTranspose2d(512, 200, 8)
        self.dBatchNorm1 = nn.BatchNorm2d(200)
        self.dConvT2 = nn.ConvTranspose2d(200, 120, 4, 4)
        self.dBatchNorm2 = nn.BatchNorm2d(120)
        self.dConvT3 = nn.ConvTranspose2d(120, 60, 2, 2)
        self.dBatchNorm3 = nn.BatchNorm2d(60)
        self.dConvT4 = nn.ConvTranspose2d(60, 30, 2, 2)
        self.dBatchNorm4 = nn.BatchNorm2d(30)
        self.dConvT5 = nn.ConvTranspose2d(30, 15, 2, 2)
        self.dBatchNorm5 = nn.BatchNorm2d(15)
        self.dConvT6 = nn.ConvTranspose2d(15, 3, 1, 1)

    def encode(self, x):
        # print('first:', x.shape)
        x = self.eConv1(x)
        # print('second:', x.shape)
        x = F.relu(x)
        x = self.eConv2(x)
        # print('third:', x.shape)
        x = F.relu(x)
        x = self.ePool1(x)
        # print('fourth:', x.shape)
        x = self.eConv3(x)
        # print('fifth:', x.shape)
        x = F.relu(x)
        x = self.ePool2(x)
        # print('sixth:', x.shape)
        x = x.view(x.size()[0], -1)
        # print('seventh:', x.shape)
        x = self.eF1(x)
        # print('eighth:', x.shape)
        mu = self.eMu(x)
        # print('mu:', mu.shape)
        sigma = self.eSigma(x)
        # print('sigma:', sigma.shape)
        return ((mu, sigma))

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return (mu + eps * std)

    def decode(self, x):
        # print('first_dec:', x.shape)
        x = torch.reshape(x, (x.shape[0], 512, 1, 1))
        # print('second_dec:', x.shape)
        x = self.dConvT1(x)
        # print('third_dec:', x.shape)
        x = self.dBatchNorm1(x)
        # print('fourth_dec:', x.shape)
        x = F.relu(x)
        x = self.dConvT2(x)
        # print('fifth_dec:', x.shape)
        x = self.dBatchNorm2(x)
        # print('sixth_dec:', x.shape)
        x = F.relu(x)
        x = self.dConvT3(x)
        # print('seventh_dec:', x.shape)
        x = self.dBatchNorm3(x)
        # print('eighth_dec:', x.shape)
        x = F.relu(x)
        x = self.dConvT4(x)
        # print('ninth_dec:', x.shape)
        x = self.dBatchNorm4(x)
        x = F.relu(x)
        # print('tenth_dec:', x.shape)
        x = self.dConvT5(x)
        # print('eleventh_dec:', x.shape)
        x = self.dBatchNorm5(x)
        # print('twelfth_dec:', x.shape)
        x = F.relu(x)
        x = self.dConvT6(x)
        # print('thirteenth_dec:', x.shape)
        x = torch.sigmoid(x)
        return (x)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_gen = self.decode(z)
        return ((x_gen, mu, sigma))


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, k, x_gen, mu, sigma):
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    BCE = criterion(x_gen, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return BCE + k * KLD