import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.benchmark as benchmark
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class EncoderBlock(nn.Module):
    """
    ResNet block, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False,transmitter_out_dim=64):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.transmitter_out_dim = transmitter_out_dim
        if self.first_conv:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.layer1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.sigmoid,
            self.maxpool
        )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.linear_enc = nn.Linear(64, self.transmitter_out_dim)
        self.linear_dec = nn.Linear(self.transmitter_out_dim,64)
        self.layer6 = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(4,512,kernel_size = 3,stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # self.layer8_downsampling = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         nn.BatchNorm2d(planes * block.expansion),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_fc = nn.Linear(512, 10)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, noise):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise

        x = F.relu(self.linear_dec(x))
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        res = self.layer8(x)
        x = self.layer9_res(res)
        x = x + res

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.final_fc(x)
        return x, output
    def get_layer_output_for_multi_exit(self,x,noise):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise

        output_0 = x

        x = F.relu(self.linear_dec(x))
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))

        output_1 = torch.flatten(x, 1)

        res = self.layer8(x)

        output_2 = torch.flatten(self.avgpool(res), 1)  # Batch, 512

        x = self.layer9_res(res)

        output_3 = torch.flatten(self.avgpool(x), 1)  # Batch, 512

        x = x + res

        output_4 = torch.flatten(self.avgpool(x), 1)  # Batch, 512

        x = self.avgpool(x)
        final_feature = torch.flatten(x, 1)
        logit = self.final_fc(final_feature)
        return final_feature, logit, [output_0, output_1, output_2, output_3, output_4]
    def get_layer_output(self, x, noise,layer_num):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise
        if layer_num == 0:
            return x                                    # 这里添加了一个layer=0的出口
        x = F.relu(self.linear_dec(x))
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        if layer_num == 1:
            # return torch.flatten(self.avgpool(x), 1)  # 这里从4维改为64维
            return torch.flatten(x,1)
        res = self.layer8(x)
        if layer_num == 2:
            return torch.flatten(self.avgpool(res),1)   # Batch, 512


        x = self.layer9_res(res)
        if layer_num == 3:
            return torch.flatten(self.avgpool(x),1)     # Batch, 512

        x = x + res
        if layer_num == 4:
            return torch.flatten(self.avgpool(x),1)     # Batch, 512

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class ResNetEncoder_change_position(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False,transmitter_out_dim=64):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.transmitter_out_dim = transmitter_out_dim
        if self.first_conv:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.layer1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.sigmoid,
            self.maxpool
        )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.linear_enc = nn.Linear(64, self.transmitter_out_dim)

        self.linear_dec_mu = nn.Linear(self.transmitter_out_dim,64)
        self.linear_dec_log_var = nn.Linear(self.transmitter_out_dim,64)

        self.layer6 = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(4,512,kernel_size = 3,stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # self.layer8_downsampling = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         nn.BatchNorm2d(planes * block.expansion),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_fc = nn.Linear(512, 10)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, noise):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise

        ####
        decx_mu = self.linear_dec_mu(x)
        decx_log_var = self.linear_dec_log_var(x)
        decx_eps = torch.randn_like(decx_mu).to('cuda' if torch.cuda.is_available() else 'cpu')
        x = decx_mu + decx_eps * decx_log_var
        ####

        # x = F.relu(self.linear_dec(x))
        x = F.relu(x)
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        res = self.layer8(x)
        x = self.layer9_res(res)
        x = x + res

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.final_fc(x)
        return x, output, [decx_mu, decx_log_var]
    def get_layer_output_for_multi_exit(self,x,noise):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise

        ####
        decx_mu = self.linear_dec_mu(x)
        decx_log_var = self.linear_dec_log_var(x)
        decx_eps = torch.randn_like(decx_mu).to('cuda' if torch.cuda.is_available() else 'cpu')
        x = decx_mu + decx_eps * decx_log_var
        ####

        output_0 = x

        # x = F.relu(self.linear_dec(x))
        x = F.relu(x)
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))

        output_1 = torch.flatten(x, 1)

        res = self.layer8(x)

        output_2 = torch.flatten(self.avgpool(res), 1)  # Batch, 512

        x = self.layer9_res(res)

        output_3 = torch.flatten(self.avgpool(x), 1)  # Batch, 512

        x = x + res

        output_4 = torch.flatten(self.avgpool(x), 1)  # Batch, 512

        x = self.avgpool(x)
        final_feature = torch.flatten(x, 1)
        logit = self.final_fc(final_feature)
        return final_feature, logit, [output_0, output_1, output_2, output_3, output_4], [decx_mu, decx_log_var]
    def get_layer_output(self, x, noise,layer_num): # only in test
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise

        ####
        decx_mu = self.linear_dec_mu(x)
        decx_log_var = self.linear_dec_log_var(x)
        decx_eps = torch.randn_like(decx_mu).to('cuda' if torch.cuda.is_available() else 'cpu')
        x = decx_mu + decx_eps * decx_log_var
        ####

        if layer_num == 0:
            return x                                    # 这里添加了一个layer=0的出口
        # x = F.relu(self.linear_dec(x))
        x = F.relu(x)
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        if layer_num == 1:
            # return torch.flatten(self.avgpool(x), 1)  # 这里从4维改为64维
            return torch.flatten(x,1)
        res = self.layer8(x)
        if layer_num == 2:
            return torch.flatten(self.avgpool(res),1)   # Batch, 512


        x = self.layer9_res(res)
        if layer_num == 3:
            return torch.flatten(self.avgpool(x),1)     # Batch, 512

        x = x + res
        if layer_num == 4:
            return torch.flatten(self.avgpool(x),1)     # Batch, 512

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class ResNetEncoder_JSCC(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False, transmitter_out_dim=None):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.transmitter_out_dim = transmitter_out_dim
        if self.first_conv:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.layer1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.sigmoid,
            self.maxpool
        )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.linear_enc = nn.Linear(64, self.transmitter_out_dim)
        self.linear_dec = nn.Linear(self.transmitter_out_dim, 64)
        self.layer6 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # self.layer8_downsampling = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         nn.BatchNorm2d(planes * block.expansion),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_fc = nn.Linear(512, 10)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, noise):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise

        x = F.relu(self.linear_dec(x))
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        res = self.layer8(x)
        x = self.layer9_res(res)
        x = x + res

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.final_fc(x)
        return output

    def get_layer_output(self, x, noise, layer_num):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise
        if layer_num == 0:
            return x                                    # 这里添加了一个layer=0的出口
        x = F.relu(self.linear_dec(x))
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        if layer_num == 1:
            return torch.flatten(self.avgpool(x),1)       # 4改成了64

        res = self.layer8(x)
        if layer_num == 2:
            return torch.flatten(self.avgpool(res), 1)

        x = self.layer9_res(res)
        if layer_num == 3:
            return torch.flatten(self.avgpool(x), 1)

        x = x + res
        if layer_num == 4:
            return torch.flatten(self.avgpool(x), 1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class ResNetEncoder_VIB(nn.Module):

    def __init__(self, block, layers, first_conv=False, maxpool1=False, transmitter_out_dim=None):
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.transmitter_out_dim = transmitter_out_dim
        if self.first_conv:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.layer1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.sigmoid,
            self.maxpool
        )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.linear_enc = nn.Linear(64, self.transmitter_out_dim)
        self.linear_dec_mu = nn.Linear(self.transmitter_out_dim, 64)
        self.linear_dec_log_var = nn.Linear(self.transmitter_out_dim, 64)
        self.layer6 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9_res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # self.layer8_downsampling = nn.Sequential(
        #         conv1x1(self.inplanes, planes * block.expansion, stride),
        #         nn.BatchNorm2d(planes * block.expansion),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_fc = nn.Linear(512, 10)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, noise):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise
        ####
        decx_mu = self.linear_dec_mu(x)
        decx_log_var = self.linear_dec_log_var(x)
        decx_eps = torch.randn_like(decx_mu).to('cuda' if torch.cuda.is_available() else 'cpu')
        x = decx_mu + decx_eps * decx_log_var
        ####

        x = F.relu(x)
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        res = self.layer8(x)
        x = self.layer9_res(res)
        x = x + res

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.final_fc(x)
        return output,[decx_mu,decx_log_var]

    def get_layer_output(self, x, noise, layer_num):
        x = self.layer1(x)
        x = self.layer2(x)  # 128,16,16
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.reshape(x, (x.size()[0], 4 * 4 * 4))
        x = self.linear_enc(x)
        x = self.sigmoid(x)

        x = x + torch.randn_like(x) * noise
        if layer_num == 0:
            return x                                    # 这里添加了一个layer=0的出口
        x = F.relu(self.linear_dec(x))
        x = torch.reshape(x, (x.size()[0], 4, 4, 4))
        if layer_num == 1:
            return torch.flatten(self.avgpool(x),1)       # 4改成了64

        res = self.layer8(x)
        if layer_num == 2:
            return torch.flatten(self.avgpool(res), 1)

        x = self.layer9_res(res)
        if layer_num == 3:
            return torch.flatten(self.avgpool(x), 1)

        x = x + res
        if layer_num == 4:
            return torch.flatten(self.avgpool(x), 1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def resnet18_encoder_semantic(first_conv, maxpool1,transmitd_out_dim):
    return ResNetEncoder(EncoderBlock, [2, 2, 2, 2, 2], first_conv, maxpool1,transmitter_out_dim=transmitd_out_dim)

def resnet18_encoder_JSCC(first_conv,maxpool1,transmit_dim):
    return ResNetEncoder_JSCC(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1,transmit_dim)
def resnet18_encoder_VIB(first_conv,maxpool1,transmit_dim):
    return ResNetEncoder_VIB(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1,transmit_dim)
def resnet18_encoder_semantic_change_position(first_conv, maxpool1,transmitter_out_dim):
    return ResNetEncoder_change_position(EncoderBlock, [2, 2, 2, 2, 2], first_conv, maxpool1, transmitter_out_dim=transmitter_out_dim)
# def resnet18_encoder_semantic_early_exit(first_conv,maxpool1):
#     return ResNetEncoder_early_exit(EncoderBlock, [2, 2, 2, 2, 2], first_conv, maxpool1, transmitter_out_dim=64)

if __name__ =='__main__':
    from torch.backends import cudnn
    import numpy as np
    import tqdm
    cudnn.benchmark = True
    model = resnet18_encoder_semantic(False,False)
    # print(model)
    device = torch.device("cuda")
    # device = torch.device("cpu")
    model.to(device)
    input_tensor = torch.randn(1,3,32,32).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)

    print('warm up ...\n')
    for _ in range(1000):
        _ = model(x,noise=0.1)
    torch.cuda.synchronize()
    # with torch.no_grad():
    #     benchmark_result = benchmark.Timer(
    #         stmt="model.get_layer_output(input_tensor,noise=0.1,layer_num=0)",
    #         globals={"model": model, "input_tensor": input_tensor},
    #         num_threads=1,
    #         # num_runs=100,
    #     ).blocked_autorange()
    # print(benchmark_result)

    repetitions = 300
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    x = torch.randn(1, 3, 32, 32).to(device)
    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model.get_layer_output(x,noise=0.1,layer_num=4)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))