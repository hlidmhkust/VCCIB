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


class semantic_backbone(nn.Module):
    def __init__(self, hidden_channel):
        super().__init__()

        self.hidden_channel = hidden_channel

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
        self.final_fc = nn.Sequential(
            nn.Linear(512, 10, bias=False),
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(64, self.hidden_channel),
            nn.Tanh()
        )

        self.decoder1_1 = nn.Sequential(
            nn.Linear(self.hidden_channel, 64),
            nn.ReLU()
        )

        self.decoder1_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.decoder1_3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Tanh = nn.Tanh()

    def encode_x(self, x):
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
        return x

    def channel_x(self, x, noise):
        x = x + torch.randn_like(x) * noise
        return x

    def forward(self, x, noise=0.1):

        x = self.encode_x(x)

        x = self.channel_x(x, noise)

        # received signal
        x = self.decoder1_1(x)  # 64,64

        x = self.decoder1_2(x)  # 64,64

        x = self.decoder1_3(x)  # 64,64

        x = torch.reshape(x, (-1, 4, 4, 4))  # 64,4,4,4

        decoded_feature = self.decoder2(x)  # 64,512,4,4

        x = self.layer3_res(decoded_feature)  # 64,512,4,4

        x = x + decoded_feature  # 64,512,4,4

        x = self.classifier1(x)  # 64,512

        output = self.final_fc(x)  # 64, 10

        return x, output

    def get_layer_output(self, x, noise, layer_num):
        x = self.encode_x(x)
        x = self.channel_x(x, noise)

        x = self.decoder1_1(x)
        x = self.decoder1_2(x)
        x = self.decoder1_3(x)

        if layer_num == 1:
            return torch.flatten(x, 1)

        x = torch.reshape(x, (-1, 4, 4, 4))  # 64,4,4,4
        if layer_num == 2:
            return torch.flatten(self.avgpool(x),1)

        decoded_feature = self.decoder2(x)  # 64,512,4,4

        x = self.layer3_res(decoded_feature)  # 64,512,4,4

        x = x + decoded_feature  # 64,512,4,4
        if layer_num == 3:
            return torch.flatten(self.avgpool(x), 1)

        x = self.classifier1(x)  # 64,512

        return x

class semantic_backbone_JSCC(nn.Module):
    def __init__(self, hidden_channel):
        super().__init__()
        self.hidden_channel = hidden_channel

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
        self.final_fc = nn.Sequential(
            nn.Linear(512, 10, bias=False),
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(64, self.hidden_channel),
            nn.Tanh()
        )

        self.decoder1_1 = nn.Sequential(
            nn.Linear(self.hidden_channel, 64),
            nn.ReLU()
        )

        self.decoder1_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.decoder1_3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Tanh = nn.Tanh()

    def encode_x(self, x):
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
        return x

    def channel_x(self, x, noise):
        x = x + torch.randn_like(x) * noise
        return x

    def forward(self, x, noise=0.1):

        x = self.encode_x(x)

        x = self.channel_x(x, noise)

        # received signal
        x = self.decoder1_1(x)  # 64,64

        x = self.decoder1_2(x)  # 64,64

        x = self.decoder1_3(x)  # 64,64

        x = torch.reshape(x, (-1, 4, 4, 4))  # 64,4,4,4

        decoded_feature = self.decoder2(x)  # 64,512,4,4

        x = self.layer3_res(decoded_feature)  # 64,512,4,4

        x = x + decoded_feature  # 64,512,4,4

        x = self.classifier1(x)  # 64,512

        output = self.final_fc(x)  # 64, 10

        return output

    def get_layer_output(self, x, noise, layer_num):
        x = self.encode_x(x)
        x = self.channel_x(x, noise)

        x = self.decoder1_1(x)
        x = self.decoder1_2(x)
        x = self.decoder1_3(x)

        if layer_num == 1:
            return torch.flatten(x, 1)

        x = torch.reshape(x, (-1, 4, 4, 4))  # 64,4,4,4
        if layer_num == 2:
            return torch.flatten(self.avgpool(x),1)

        decoded_feature = self.decoder2(x)  # 64,512,4,4

        x = self.layer3_res(decoded_feature)  # 64,512,4,4

        x = x + decoded_feature  # 64,512,4,4
        if layer_num == 3:
            return torch.flatten(self.avgpool(x), 1)

        x = self.classifier1(x)  # 64,512

        return x
if __name__ == '__main__':
    hidden_channel = 64


    model = semantic_backbone_JSCC(hidden_channel).cuda()
    x = torch.randn(7,3,32,32).cuda()

    l_feature = [torch.flatten(model.get_layer_output(x,0.1,i),1) for i in range(1,5)]
    l_feature = torch.hstack(l_feature)
    a=1
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # trainset = torchvision.datasets.CIFAR10(root='/home/hlidm/Datasets/cifar10', train=True, download=True,
    #                                         transform=transform_train)
    # testset = torchvision.datasets.CIFAR10(root='/home/hlidm/Datasets/cifar10', train=False, download=True,
    #                                        transform=transform_test)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False, num_workers=2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # for epoch in range(0, 100):
    #     for i, (x, y) in enumerate(train_loader):
    #         x = x.cuda()
    #         y = y.cuda()
    #         model.train()
    #         output = model(x, noise=0.1)[1]
    #         criterion = nn.CrossEntropyLoss().cuda()
    #         loss1 = criterion(output, y)
    #         optimizer.zero_grad()
    #         loss1.backward()
    #         optimizer.step()
    #     print(f"epoch:{epoch}loss:{loss1.item()}")
    #     with torch.no_grad():
    #         model.eval()
    #         correct = 0
    #         total = 0
    #         for i, (x, y) in enumerate(test_loader):
    #             x = x.cuda()
    #             y = y.cuda()
    #             output = model(x, noise=0.1)[1]
    #             _, pred = torch.max(output, 1)
    #             total += y.size(0)
    #             correct += (pred == y).sum().item()
    #         print("Test Acc:", 100 * correct / total)
