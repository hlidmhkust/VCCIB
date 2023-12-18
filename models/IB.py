import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import argparse
import copy


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))


class IB_model(nn.Module):
    def __init__(self, args):
        super(IB_model, self).__init__()
        self.config = args

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer1_res = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.layer3_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.classifier1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
            Flatten()
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(512, 10, bias=False),
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.encoder3 = nn.Linear(64, self.config.enc_out_dim)

        self.decoder_fc_mu = nn.Linear(self.config.enc_out_dim, self.config.latent_dim)
        self.decoder_fc_logvar = nn.Linear(self.config.enc_out_dim, self.config.latent_dim)
        self.decoder1_3 = nn.Sequential(
            nn.Linear(self.config.latent_dim, 64),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.Tanh = nn.Tanh()

    def encode_channel_x(self, x, noise):
        channel_noise = noise
        x = self.prep(x)
        x = self.layer1(x)
        res = self.layer1_res(x)
        x = res + x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.encoder1(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = torch.tanh(x)

        # channel part
        x = x + torch.randn_like(x) * channel_noise
        return x

    def forward(self, x, noise):
        x = self.encode_channel_x(x, noise)
        # decoder part
        mu = self.decoder_fc_mu(x)
        log_var = self.decoder_fc_logvar(x)
        eps = torch.randn_like(log_var)
        std = torch.exp(log_var / 2)
        z = mu + eps * std
        KL = self.KLD(mu, log_var)

        x = self.decoder1_3(z)
        x = torch.reshape(x, (-1, 4, 4, 4))
        decoded_feature = self.decoder2(x)
        x = self.layer3_res(decoded_feature)
        x = x + decoded_feature
        x = self.classifier1(x)
        output = self.classifier2(x)

        return output, KL, mu, log_var

    def KLD(self, z_mu, z_logvar):
        # between N(0,I) gaussian and z gaussian
        kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
        return kld_loss / z_mu.size(0)

    def get_layer_output(self, x, noise, layer_num):
        x = self.encode_channel_x(x, noise)
        mu = self.decoder_fc_mu(x)
        log_var = self.decoder_fc_logvar(x)
        eps = torch.randn_like(log_var)
        std = torch.exp(log_var / 2)
        z = mu + eps * std

        x = self.decoder1_3(z)
        if layer_num == 1:
            return torch.flatten(self.avgpool(x), 1)
        x = torch.reshape(x, (-1, 4, 4, 4))

        decoded_feature = self.decoder2(x)
        if layer_num == 2:
            return torch.flatten(self.avgpool(decoded_feature), 1)

        x = self.layer3_res(decoded_feature)
        if layer_num == 3:
            return torch.flatten(self.avgpool(x), 1)

        x = x + decoded_feature
        x= self.classifier1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == '__main__':
    x = torch.randn(7, 3, 32, 32).cuda()


    class args:
        enc_out_dim = 512
        latent_dim = 17


    noise = 0.0352
    model = IB_model(args).cuda()
    y = model(x, noise)
