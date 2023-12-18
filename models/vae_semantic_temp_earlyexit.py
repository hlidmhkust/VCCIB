from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR,CyclicLR
from pytorch_metric_learning import losses,miners,distances
from torchvision.transforms.transforms import RandomRotation
from losses.modded_triplets import TripletMarginLossModded
from pytorch_lightning.callbacks import Callback
from sklearn.decomposition import PCA
from pl_bolts.models.autoencoders import AE
from models.mnist_ae import MNIST_AE
# import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
# try:
#     from tsnecuda import TSNE
# except:
from sklearn.manifold import TSNE
import numpy as np
from pytorch_lightning.loggers.neptune import NeptuneLogger
from sklearn.metrics import f1_score,auc
from models.decoders_encoders import *
from tqdm import tqdm
from neptunecontrib.api import log_table
from losses.losses import SupConLoss
from losses.mmd import compute_mmd
from PIL import Image
from torchvision.models import vgg11
import kornia
from torchvision import transforms
from models.densenet import *
from models.vgg import VGGM
import seaborn as sns
from pl_bolts.datamodules import CIFAR10DataModule
from models.wide_resnet import WideResNet
import matplotlib.ticker as ticker
from models.resnet_encoders_semantic_temp import resnet18_encoder_semantic
import pathlib
from time import time
from models.vae_semantic_temp import GMM_VAE_Contrastive
class UpdateMargin(Callback):
    def __init__(self,margin_max_distance, margin_jumps,margins_epoch):
        self.margin_max_distance = margin_max_distance
        self.margin_jumps = margin_jumps
        self.margins_epoch = margins_epoch

    def on_train_epoch_start(self,trainer, pl_module):
        if trainer.current_epoch >= self.margins_epoch and pl_module.metric_loss.neg_margin <= self.margin_max_distance:
            pl_module.metric_loss.neg_margin +=self.margin_jumps



class InitializeGaussians(Callback):
    def on_test_epoch_start(self,trainer,pl_module):
        pl_module.eval()
        if trainer.current_epoch == 0:
            pl_module.initialize_gaussians()

ae_dict = {
    'cifar_known':(lambda:AE(input_height=32), 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    'cifar100_known':(lambda:AE(input_height=32,enc_type='resnet50'), 'models/ae_cifar100.ckpt'),
    'mnist':(lambda:MNIST_AE(input_height=28,enc_type='mnist'),'models/ae_mnist.ckpt')
}
# backbone_dict = {
# 'resnet18_semantic_temp':(lambda:GMM_VAE_Contrastive(input_height=32,enc_type='resnet18_semantic_temp',first_conv=False,maxpool1=False,enc_out_dim=512,kl_coeff=0.1,latent_dim=32,lr=0.01,class_num=10,
#                                                            generation_epoch=0,lower_bound=0.95,step_size=50,step_gamma=0.1,cov_scaling=None,recon_weight=None,gen_weight=None,log_tsne=None,weights=None,finetune=None,
#                                                            finetune_lr=None,known_dataset='cifar_known',unknown_dataset='imagenet-resize',opt='sgd',ae_features=False,margin_max_distance=32.0,sample_count=8,noise=0.1),
#                                      './checkpoints/resnet18_semantic_temp/cifar_known/version_2/checkpoint-epoch=475-f1_score=0.789.ckpt'),
# }
class GMM_VAE_Contrastive_early_exit(pl.LightningModule):
    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18_semantic_temp_early_exit',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        class_num: int = 10,
        generation_epoch: int = 50,
        lower_bound: float = 0.05,
        step_size: int = 100,
        step_gamma: float = 0.1,
        cov_scaling: float = 5.0,
        recon_weight: float = 0.1,
        gen_weight: float = 5.0,
        log_tsne:bool = False,
        weights:str = None,
        finetune:bool = False,
        finetune_lr:float = 0.001,
        known_dataset:str = 'cifar_known',
        unknown_dataset:str = None,
        opt:str ='sgd',
        ae_features: bool= False,
        margin_max_distance: float = 32,
        sample_count: int = 40,
        noise:float = None,
        transmit_dim:int = None,
        backbone_type = None,
        backbone_checkpoint = None,
        mode = None,
        **kwargs
    ):

        super(GMM_VAE_Contrastive_early_exit, self).__init__()

        self.save_hyperparameters()
        self.known_dataset = known_dataset
        self.unknown_dataset = unknown_dataset
        self.lr = lr
        self.mode = mode
        self.transmit_dim = transmit_dim
        self.enc_type = enc_type
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.class_num = class_num
        self.generation_epoch = generation_epoch
        self.log_tsne = log_tsne
        self.is_tested = 0
        self.criterion = nn.CrossEntropyLoss()
        self.opt = opt
        self.ae_features = ae_features
        self.margin_max_distance = margin_max_distance
        self.sample_count = sample_count
        self.noise = noise
        self.backbone_type = backbone_type
        self.backbone_checkpoint = backbone_checkpoint
        self.metric_loss = TripletMarginLossModded(margin=0.1,neg_margin=0.15)
        self.precentiles = []
        self.class_gaussians = None
        self.lower_bound = lower_bound
        self.ae = None
        self.backbone = None
        self.backbone_dict = {
            'resnet18_semantic_temp': (
            lambda: GMM_VAE_Contrastive(input_height=32, enc_type='resnet18_semantic_temp', first_conv=False,
                                        maxpool1=False, enc_out_dim=512, kl_coeff=0.1, latent_dim=32, lr=0.01,
                                        class_num=10,
                                        generation_epoch=0, lower_bound=0.95, step_size=50, step_gamma=0.1,
                                        cov_scaling=None, recon_weight=None, gen_weight=None, log_tsne=None,
                                        weights=None, finetune=None,
                                        finetune_lr=None, known_dataset='cifar_known',
                                        unknown_dataset='imagenet-resize', opt='sgd', ae_features=False,
                                        margin_max_distance=32.0, sample_count=8, noise=self.noise,transmit_dim=self.transmit_dim),
            './checkpoints/resnet18_semantic_temp/cifar_known/version_2/checkpoint-epoch=475-f1_score=0.789.ckpt'),
        }
        self.initilize_backbone()

        self.fc_mu_exit0 = nn.Linear(64,self.latent_dim)
        self.fc_var_exit0 = nn.Linear(64,self.latent_dim)
        self.final_fc_exit0 = nn.Linear(self.latent_dim,self.class_num)

        self.fc_mu_exit1 = nn.Linear(64,self.latent_dim)
        self.fc_var_exit1 = nn.Linear(64,self.latent_dim)
        self.final_fc_exit1 = nn.Linear(self.latent_dim, self.class_num)

        self.fc_mu_exit2 = nn.Linear(512, self.latent_dim)
        self.fc_var_exit2 = nn.Linear(512, self.latent_dim)
        self.final_fc_exit2 = nn.Linear(self.latent_dim, self.class_num)

        self.fc_mu_exit3 = nn.Linear(512, self.latent_dim)
        self.fc_var_exit3 = nn.Linear(512, self.latent_dim)
        self.final_fc_exit3 = nn.Linear(self.latent_dim, self.class_num)

        self.fc_mu_exit4 = nn.Linear(512, self.latent_dim)
        self.fc_var_exit4 = nn.Linear(512, self.latent_dim)
        self.final_fc_exit4 = nn.Linear(self.latent_dim, self.class_num)
    def initilize_ae(self):
        self.ae = ae_dict['cifar_known'][0]()
        self.ae = self.ae.load_from_checkpoint(ae_dict['cifar_known'][1])
        self.ae.cuda()
        self.ae.freeze()
    def initilize_backbone(self):
        #!!!!!!!!!!!!  可能有问题
        self.backbone = self.backbone_dict[self.backbone_type][0]()
        self.backbone.cuda()
        self.backbone = self.backbone.load_from_checkpoint(self.backbone_dict[self.backbone_type][1])
        self.backbone.transmit_dim = self.transmit_dim
        print(self.backbone_checkpoint)
        # self.backbone.load_from_checkpoint(self.backbone_checkpoint)
        self.backbone.freeze()

    def forward(self, x):
        x = self.encoder(x,self.noise)
        # return x
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z

    def _run_step(self, x):
        x,logit = self.encoder(x,self.noise)
        # return x
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        # return z,q
        return z, mu, log_var, q,logit

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def reparam_sample(self,mu,logvar):
        # for early_exit
        std= torch.exp(logvar/2)
        eps = torch.randn_like(mu)
        r_sample = mu+std*eps
        return r_sample

    # 浓缩，只剩下每个类别都不属于的数据
    def filter_samples(self,all_samples,labels):
        for i in range(self.class_num):
            probs = self.class_gaussians[i].log_prob(all_samples) < self.class_bounds[i]
            all_samples = all_samples[probs]
            labels = labels[probs]
        return all_samples,labels

    def filter_just_samples(self,samples):
        for i in range(self.class_num):
            probs = self.class_gaussians[i].log_prob(samples) < self.class_bounds[i]
            samples = samples[probs]
        return samples




    # def step(self, batch, batch_idx):
    #     x, y = batch
    #     cur_classes = torch.unique(y).long()
    #
    #     z ,mu,log_var, q,logit= self._run_step(x)
    #
    #     kl = 0
    #     log_qz = q.log_prob(z)
    #     for i in cur_classes:
    #
    #         y_count = (y==i).sum()
    #         rel_zs = z[y==i,:]
    #         self.class_temp_means[i,:] += rel_zs.sum(axis = 0).detach()
    #         self.class_temp_cov[i,:,:] += torch.matmul(rel_zs.T,rel_zs).detach()
    #         self.class_counts[i] += y_count.detach()
    #         #Estimation:
    #         cp = torch.distributions.Normal(self.class_means[i,:].repeat(y_count,1), torch.ones_like(rel_zs)*(0.1 if i < self.class_num else 1))
    #         log_pz = cp.log_prob(rel_zs)
    #         kl_class = log_qz[y==i] - log_pz
    #         kl_class = kl_class.mean()
    #         kl_class *= self.kl_coeff
    #         kl += kl_class / len(cur_classes)
    #         #Analytical
    #         # rel_mu = mu[y==i,:]
    #         # rel_var = torch.exp(log_var[y==i,:]/2)
    #         # q = torch.distributions.Normal(rel_mu, rel_var)
    #         # p = torch.distributions.Normal(self.class_means[i,:].repeat(y_count,1), torch.ones_like(rel_zs)*(0.1 if i < self.class_num else 1))
    #         # kl_loss = torch.distributions.kl.kl_divergence(q,p).mean()*self.kl_coeff
    #         # kl += kl_loss / len(cur_classes)
    #         if self.trainer.current_epoch > 0:
    #             class_ll = self.class_gaussian[i].log_prob(rel_zs).detach()
    #             self.precentiles[i].append(class_ll)
    #
    #     contrastive_loss = self.metric_loss(z,y)
    #     ce_loss = self.criterion(logit,y)
    #     if self.trainer.current_epoch < self.generation_epoch:
    #         kl = 0
    #     loss = ce_loss + contrastive_loss + kl
    #     logs = {
    #         "ce_loss":ce_loss,
    #         "loss": loss,
    #         "kl": kl,
    #         "triplet_loss": contrastive_loss,
    #     }
    #     return loss, logs



    def on_test_epoch_start(self) -> None:
        self.initilize_ae()
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.enc_type == 'resnet18_semantic_temp_early_exit':
            final_mu, final_var, logit, early_features = self.backbone.final_distribution_logit(x, self.noise)
            mu_0 = self.fc_mu_exit0(early_features[0])
            mu_1 = self.fc_mu_exit1(early_features[1])
            mu_2 = self.fc_mu_exit2(early_features[2])
            mu_3 = self.fc_mu_exit3(early_features[3])
            mu_4 = self.fc_mu_exit4(early_features[4])
            z = [mu_0, mu_1, mu_2, mu_3, mu_4]
        else:
            raise Exception("Please check enc_type")

        l_features_0 = [torch.flatten(self.backbone.encoder.get_layer_output(x,self.noise,i),1) for i in range(0,1)]  + [z[0]]
        l_features_0 = torch.hstack(l_features_0)

        l_features_1 = [torch.flatten(self.backbone.encoder.get_layer_output(x, self.noise, i), 1) for i in range(0, 2)] + [z[1]]
        l_features_1 = torch.hstack(l_features_1)

        l_features_2 = [torch.flatten(self.backbone.encoder.get_layer_output(x, self.noise, i), 1) for i in range(0, 3)] + [z[2]]
        l_features_2 = torch.hstack(l_features_2)

        l_features_3 = [torch.flatten(self.backbone.encoder.get_layer_output(x, self.noise, i), 1) for i in range(0, 4)] + [z[3]]
        l_features_3 = torch.hstack(l_features_3)

        l_features_4 = [torch.flatten(self.backbone.encoder.get_layer_output(x, self.noise, i), 1) for i in range(0, 5)] + [z[4]]
        l_features_4 = torch.hstack(l_features_4)

        l_features=[l_features_0,l_features_1,l_features_2,l_features_3,l_features_4]

        cur_classes = torch.unique(y)
        if self.trainer.testing and self.is_tested == 0:
            for i in cur_classes:
                y_count = (y == i).sum()
                self.class_counts[i] += y_count.detach()
            for i in cur_classes:
                y_count = (y==i).sum()

                rel_l_features_0 = l_features_0[y==i,:]
                self.early_temp_means_0[i, :] = rel_l_features_0.sum(axis=0).detach()
                self.early_temp_cov_0[i, :, :] = torch.matmul(rel_l_features_0.T, rel_l_features_0).detach()

                rel_l_features_1 = l_features_1[y == i, :]
                self.early_temp_means_1[i, :] = rel_l_features_1.sum(axis=0).detach()
                self.early_temp_cov_1[i, :, :] = torch.matmul(rel_l_features_1.T, rel_l_features_1).detach()

                rel_l_features_2 = l_features_2[y == i, :]
                self.early_temp_means_2[i, :] = rel_l_features_2.sum(axis=0).detach()
                self.early_temp_cov_2[i, :, :] = torch.matmul(rel_l_features_2.T, rel_l_features_2).detach()

                rel_l_features_3 = l_features_3[y == i, :]
                self.early_temp_means_3[i, :] = rel_l_features_3.sum(axis=0).detach()
                self.early_temp_cov_3[i, :, :] = torch.matmul(rel_l_features_3.T, rel_l_features_3).detach()

                rel_l_features_4 = l_features_4[y == i, :]
                self.early_temp_means_4[i, :] = rel_l_features_4.sum(axis=0).detach()
                self.early_temp_cov_4[i, :, :] = torch.matmul(rel_l_features_4.T, rel_l_features_4).detach()


        x_hat=x
        if self.is_tested != 0:
            xshape = x.shape[1]
            chunk_size = int(x.shape[0] / 4)
            x_hat = []
            for i in range(4):
                x_hat.append(self.ae(x[chunk_size * i:chunk_size * (i + 1)]))   #防止out of memory
            x_hat = torch.vstack(x_hat)
            desc_x_encoder = torch.flatten(self.backbone.encoder.get_layer_output(x, self.noise,1), 1)
            desc_x_hat_encoder = torch.flatten(self.backbone.encoder.get_layer_output(x_hat,self.noise, 1), 1)
            desc_x = desc_x_encoder
            desc_x_hat = desc_x_hat_encoder
        else:
            desc_x = 0
            desc_x_hat = 0
            x_hat = x
        if self.enc_type == 'resnet18_semantic_temp_early_exit':
            return z,y,x_hat,x,desc_x,desc_x_hat,l_features
    def training_epoch_end(self, outputs):
        pass

    def kl_divergence_for_gaussian(self, q_batch_mu, q_batch_var, p_batch_mu, p_batch_var):
        # between N(mu_1,sigma^2_1) and N(mu_2,sigma^2_2)
        # D(q||p) = q log(q/p)
        term1 = torch.sum(0.5 * (p_batch_var - q_batch_var))
        term2 = torch.sum(
            (torch.exp(q_batch_var) + torch.pow(q_batch_mu - p_batch_mu, 2)) / (2 * torch.exp(p_batch_var)) - 0.5)
        return (term1 + term2) / q_batch_var.size(0)

    def kl_divergence_discrete_distribution(self,q_prob, p_prob):
        return torch.sum(q_prob * torch.log(q_prob / p_prob))/q_prob.size(0)

    def KLD(self, z_mu, z_logvar):
        # between N(0,I) gaussian and z gaussian
        # D(q||N(0,I)) = q log(q/N(0,I))
        kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
        return kld_loss / z_mu.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        final_mu,final_var,logit,early_features = self.backbone.final_distribution_logit(x,self.noise)
        target_prob = F.softmax(logit, dim=1)

        mu_0 = self.fc_mu_exit0(early_features[0])
        log_var_0 = self.fc_var_exit0(early_features[0])
        r_sample_0 = self.reparam_sample(mu_0,log_var_0)
        exit_prob_0 = F.softmax(self.final_fc_exit0(r_sample_0),dim=1)
        loss0 = self.kl_divergence_for_gaussian(mu_0,log_var_0,final_mu,final_var)+self.kl_divergence_discrete_distribution(exit_prob_0,target_prob)

        mu_1 = self.fc_mu_exit1(early_features[1])
        log_var_1 = self.fc_var_exit1(early_features[1])
        r_sample_1 = self.reparam_sample(mu_1, log_var_1)
        exit_prob_1 = F.softmax(self.final_fc_exit1(r_sample_1), dim=1)
        loss1 = self.kl_divergence_for_gaussian(mu_1, log_var_1, final_mu,final_var) + self.kl_divergence_discrete_distribution(exit_prob_1,target_prob)

        mu_2 = self.fc_mu_exit2(early_features[2])
        log_var_2 = self.fc_var_exit2(early_features[2])
        r_sample_2 = self.reparam_sample(mu_2, log_var_2)
        exit_prob_2 = F.softmax(self.final_fc_exit2(r_sample_2), dim=1)
        loss2 = self.kl_divergence_for_gaussian(mu_2, log_var_2, final_mu,final_var) + self.kl_divergence_discrete_distribution(exit_prob_2,target_prob)

        mu_3 = self.fc_mu_exit3(early_features[3])
        log_var_3 = self.fc_var_exit3(early_features[3])
        r_sample_3 = self.reparam_sample(mu_3, log_var_3)
        exit_prob_3 = F.softmax(self.final_fc_exit3(r_sample_3), dim=1)
        loss3 = self.kl_divergence_for_gaussian(mu_3, log_var_3, final_mu,final_var) + self.kl_divergence_discrete_distribution(exit_prob_3,target_prob)

        mu_4 = self.fc_mu_exit4(early_features[4])
        log_var_4 = self.fc_var_exit4(early_features[4])
        r_sample_4 = self.reparam_sample(mu_4, log_var_4)
        exit_prob_4 = F.softmax(self.final_fc_exit4(r_sample_4), dim=1)
        loss4 = self.kl_divergence_for_gaussian(mu_4, log_var_4, final_mu,final_var) + self.kl_divergence_discrete_distribution(exit_prob_4, target_prob)

        loss = loss0+loss1+loss2+loss3+loss4
        logs = {
            "loss0":loss0,
            "loss1":loss1,
            "loss2":loss2,
            "loss3":loss3,
            "loss4":loss4
        }
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # z,y,x_hat,x,_,_,_,_ = self.test_step(batch, batch_idx)
        # return z,y,x_hat,x
        return None
    def classify_data(self,data):
        parr = torch.zeros(data.shape[0],self.class_num)
        for i in range(self.class_num):
            probs = self.class_gaussians[i].log_prob(data)
            probs[probs < self.class_bounds[i]] = -10**10
            parr[:,i] = probs
        values,labels = torch.max(parr,dim=1)
        labels[values < -10**9] = self.class_num
        print(len(labels))
        print((labels == self.class_num).sum())
        return labels.to(self.device)


    def get_data_top_val(self,data,all_sift_distances,all_labels,exit):
        parr = torch.zeros(data.shape[0],self.class_num)
        for i in range(self.class_num):
            probs = self.class_gaussians[exit][i].log_prob(data)
            parr[:,i] = probs
        max_num = parr.max()+1
        for i in range(self.class_num):
            probs = self.class_gaussians[exit][i].log_prob(data)
            probs -= max_num
            parr[:,i] = probs*(all_sift_distances)
        # parr[:,self.class_num] /= 1000
        # row_sum = torch.logsumexp(parr,1)
        values,labels = torch.max(parr,dim=1)   # range: [-max ~ 0]
        temp_values = values/torch.max(torch.abs(values))

        top2_values, top2_lables = torch.topk(parr,k=2,dim=1)
        top2_values = top2_values.transpose(0,1)
        top2_lables = top2_lables.transpose(0,1)
        second_values, second_labels = top2_values[1], top2_lables[1]
        temp_second_values = second_values/torch.max(torch.abs(values)) #这个地方可能还是不在[0,1]区间内，原因是前面的parr已经经过all_sift_distance放缩过了，后面可以将confidence score crop到[0,1]区间内

        confidence_score = (temp_values-temp_second_values)
        return temp_values,labels,second_values,second_labels,confidence_score

    def get_fpr_tpr(self,values,pred_labels,labels,threshold):
        neg_samples_count = float(np.sum(labels == self.class_num))
        pos_samples_count = float(np.sum(labels < self.class_num))
        neg_vals = values[labels == self.class_num]
        pos_vals = values[labels < self.class_num]
        pos_samples_count_cor = float(np.sum(pred_labels[labels < self.class_num] == labels[labels < self.class_num]))  #id预测为id并且exact match,9248
        fp_count = float(np.sum(neg_vals >= threshold)) #大于threshold,于是认为是positive，但实际上是negative(从neg_val里筛选出来的)
        tp_count_cor = float(np.sum((pos_vals >= threshold)& (pred_labels[labels < self.class_num] == labels[labels < self.class_num])))# id预测为id并且exact match，同时把握大于threshold
        tp_count = float(np.sum(pos_vals >= threshold)) #大于threshold,于是认为是positve，而实际上也是postive
        return fp_count/neg_samples_count,\
               tp_count/pos_samples_count,\
               tp_count_cor/pos_samples_count_cor, tp_count_cor/pos_samples_count
        # fp tp tp_cor

    def log_test_statistics(self,values,pred_labels,labels,exit):
        print(np.min(values),np.max(values))
        thresholds = np.flip(np.arange(np.min(values),np.max(values),1.0))
        thresholds = np.flip(np.arange(-1000,0,1))#根据不同的thresbold将已经预测为id数据的再划归到ood里
        f1_scores = []
        pos_acc = []
        neg_acc = []
        pos_labels = labels[labels < self.class_num]
        neg_labels = labels[labels == self.class_num]
        pos_pred_labels = pred_labels[labels < self.class_num]
        neg_pred_labels = pred_labels[labels == self.class_num]
        self.log(f"Acc Pos (no unkown det) exit_{exit}:", float(np.sum(pos_pred_labels == pos_labels)/len(pos_labels)))
        misclassification_as_openset = []
        for t in tqdm(thresholds):
            cur_preds = np.copy(pred_labels)
            cur_preds[values < t] = self.class_num

            pn = float(np.sum(cur_preds == self.class_num))
            pp = float(np.sum(cur_preds < self.class_num))

            tn = float(np.sum(labels[cur_preds == self.class_num] == self.class_num))
            fn = float(np.sum(labels[cur_preds == self.class_num] < self.class_num))
            tp = float(np.sum(labels[cur_preds < self.class_num] < self.class_num))
            fp = float(np.sum(labels[cur_preds < self.class_num] == self.class_num))
            f1_scores.append(f1_score(cur_preds,labels,average='macro'))

            pos_pred_labels = cur_preds[labels < self.class_num]

            pos_acc.append(tp/len(pos_labels))
            neg_acc.append(tn/len(neg_labels))
            mistakes = float(np.sum(pos_pred_labels == self.class_num))/ float(np.sum(pos_pred_labels != pos_labels))
            misclassification_as_openset.append(mistakes)
        f1_scores = np.array(f1_scores)
        pos_acc = np.array(pos_acc)
        neg_acc = np.array(neg_acc)
        best_t = np.argmax(f1_scores)
        self.log(f"Exit:{exit}, Best f1 score (threshold:{thresholds[best_t]}) mistakes_as_open:{misclassification_as_openset[best_t]}:", np.max(f1_scores))
        fig = plt.figure()
        plt.plot(thresholds, f1_scores, color='darkorange')
        plt.xlim([np.min(thresholds), np.max(thresholds)])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.title(f'F1 per threshold_exit{exit}')
        # self.trainer.logger.experiment.log_image('F1 Curve', fig, description=f'F1 Curve')
        self.trainer.logger.experiment.add_figure('F1 Curve', fig)
        df = pd.DataFrame()
        df['Threshold'] = thresholds
        df['F1'] = f1_scores
        df['Pos acc'] = pos_acc
        df['Neg acc'] = neg_acc
        df.to_csv(f'output/{self.enc_type}/{self.mode}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}/Threshold.csv')




    def plot_roc_curve_each_exit(self,all_data,labels,all_sift_distances,exit):
        all_data = torch.tensor(all_data,device=self.device)
        values,pred_labels,second_values,second_labels,confidence_score  = self.get_data_top_val(all_data,all_sift_distances,labels,exit)#这里会减掉最大的logit所以ood的数据负的更多
        values = values.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        self.log_test_statistics(values,pred_labels,labels,exit)
        thresholds = np.flip(np.geomspace(np.max(values),np.min(values),num=100000))#geomspace 取对数后构成等差数列
        fpr = []
        tpr = []
        acc_list = []
        aurocNew = 0.0
        fprTemp = 1.0
        fp=0
        tp=0
        tp_cor=0
        fpr_at_095 = 0
        tpr_at_095 = 0


        for t in tqdm(range(len(thresholds))):
            fp,tp,tp_cor,acc = self.get_fpr_tpr(values,pred_labels,labels,thresholds[t])#tp_cor
            # both fp and tp is decreascing
            if tp >= 0.95:
                fpr_at_095 = fp
                tpr_at_095 = tp
                acc_at_095 = acc
            if fp >= 1-0.9: # tn <= 0.9
                fpr_at_tn_090 = fp
                tpr_at_tn_090 = tp
                acc_at_tn_090 = acc

            fpr.append(fp)
            tpr.append(tp)
            acc_list.append(acc)
            aurocNew += (-fp+fprTemp)*tp_cor
            fprTemp = fp
        aurocNew += fp * tp_cor
        self.log(f'CAUROC-{exit}',aurocNew)
        self.log(f"fpr-at-tpr-095-{exit}",fpr_at_095)
        self.log(f"tpr-at-tpr-095-{exit}",tpr_at_095)
        self.log(f"tnr-at-tpr-095-{exit}",1-fpr_at_095)
        self.log(f"acc-at-tpr-095-{exit}",acc_at_095)

        self.log(f"fpr-at-tnr-090-{exit}",fpr_at_tn_090)
        self.log(f"tnr-at-tnr-090-{exit}",1-fpr_at_tn_090)
        self.log(f"tpr-at-tnr-090-{exit}",tpr_at_tn_090)
        self.log(f"acc-at-tnr-090-{exit}",acc_at_tn_090)
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
        plt.xlim([-0.00099, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Roc Curve-{exit}')
        # self.trainer.logger.experiment.log_image('ROC', fig, description=f'ROC TEST')
        self.trainer.logger.experiment.add_figure(f'ROC-{exit}', fig)
        self.log(f'ROC-AUC-Score-{exit}',auc(fpr,tpr))
        df = pd.DataFrame()
        df['Threshold'] = thresholds
        df['FPR'] = np.array(fpr)
        df['TPR'] = np.array(tpr)
        df['acc'] = np.array(acc_list)
        # log_table('FPR_TPR', df,experiment=self.trainer.logger.experiment) #这个也没法改
        df.to_csv(f'output/{self.enc_type}/{self.mode}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}/FPR_TPR-{exit}.csv')
    def plot_roc_curve_multi_exit(self,all_data,labels,all_sift_distances):
        all_data = [torch.tensor(exit_data,device=self.device) for exit_data in all_data ]
        value_list = []
        pred_labels_list = []
        second_values_list = []
        second_labels_list = []
        confidence_score_list = []
        for exit in range(0, 5):
            values,pred_labels,second_values,second_labels,confidence_score  = self.get_data_top_val(all_data[exit],all_sift_distances,labels,exit)#这里会减掉最大的logit所以ood的数据负的更多
            value_list.append(values)
            pred_labels_list.append(pred_labels)
            second_values_list.append(second_values)
            second_labels_list.append(second_labels)
            confidence_score_list.append(confidence_score)
        value_multi_exit = torch.stack(value_list).transpose(0,1)
        pred_labels_multi_exit = torch.stack(pred_labels_list).transpose(0,1)
        second_values_multi_exit = torch.stack(second_values_list).transpose(0,1)
        second_labels_multi_exit = torch.stack(second_labels_list).transpose(0,1)
        confidence_score_multi_exit = torch.stack(confidence_score_list).transpose(0,1)

        # 在所有confidence都不小于candidate情况下，把confidence最小的值都赋值给最后一个出口，有可能是他自己
        no_threshold_constrain_where_to_output = torch.min(confidence_score_multi_exit, dim=1)[1]
        no_threshold_constrain_where_to_output_confidence = torch.min(confidence_score_multi_exit, dim=1)[0]
        no_threshold_constrain_where_to_output_values = value_multi_exit[torch.arange(confidence_score.shape[0]), no_threshold_constrain_where_to_output]
        no_threshold_constrain_where_to_output_pred_labels = pred_labels_multi_exit[torch.arange(confidence_score.shape[0]), no_threshold_constrain_where_to_output]

        value_multi_exit[:,4] = no_threshold_constrain_where_to_output_values
        pred_labels_multi_exit[:,4] = no_threshold_constrain_where_to_output_pred_labels


        output_threshold_candidate = [0.05,0.1,0.15,0.2]
        for candidate in output_threshold_candidate:
            temp_confidence = confidence_score_multi_exit
            temp_confidence[:][4]=-1
            threshold_constrain_where_to_output = temp_confidence < candidate
            where_to_output = torch.argmax(threshold_constrain_where_to_output.ne(0)*1.0,dim=1)
            where_to_output_counts = torch.bincount(where_to_output, minlength=5)
            print(f'candidate:{candidate},{where_to_output_counts}')
            where_to_output_values = torch.gather(value_multi_exit, 1, where_to_output.unsqueeze(1)).squeeze()
            where_to_output_labels = torch.gather(pred_labels_multi_exit, 1, where_to_output.unsqueeze(1)).squeeze()
            values = where_to_output_values.cpu().numpy()
            pred_labels = where_to_output_labels.cpu().numpy()
            self.log_test_statistics(values,pred_labels,labels,exit)
            thresholds = np.flip(np.geomspace(np.max(values),np.min(values),num=100000))#geomspace 取对数后构成等差数列
            fpr = []
            tpr = []
            acc_list = []
            aurocNew = 0.0
            fprTemp = 1.0
            fp=0
            tp=0
            tp_cor=0
            fpr_at_095 = 0
            tpr_at_095 = 0


            for t in tqdm(range(len(thresholds))):
                fp,tp,tp_cor,acc = self.get_fpr_tpr(values,pred_labels,labels,thresholds[t])#tp_cor
                # both fp and tp is decreascing
                if tp >= 0.95:
                    fpr_at_095 = fp
                    tpr_at_095 = tp
                    acc_at_095 = acc
                if fp >= 1-0.9: # tn <= 0.9
                    fpr_at_tn_090 = fp
                    tpr_at_tn_090 = tp
                    acc_at_tn_090 = acc

                fpr.append(fp)
                tpr.append(tp)
                acc_list.append(acc)
                aurocNew += (-fp+fprTemp)*tp_cor
                fprTemp = fp
            aurocNew += fp * tp_cor
            self.log(f'CAUROC-multi-exit-{candidate}',aurocNew)
            self.log(f"fpr-at-tpr-095-multi-exit-{candidate}",fpr_at_095)
            self.log(f"tpr-at-tpr-095-multi-exit-{candidate}",tpr_at_095)
            self.log(f"tnr-at-tpr-095-multi-exit-{candidate}",1-fpr_at_095)
            self.log(f"acc-at-tpr-095-multi-exit-{candidate}",acc_at_095)

            self.log(f"fpr-at-tnr-090-multi-exit-{candidate}",fpr_at_tn_090)
            self.log(f"tnr-at-tnr-090-multi-exit-{candidate}",1-fpr_at_tn_090)
            self.log(f"tpr-at-tnr-090-multi-exit-{candidate}",tpr_at_tn_090)
            self.log(f"acc-at-tnr-090-multi-exit-{candidate}",acc_at_tn_090)
            fig = plt.figure()
            plt.plot(fpr, tpr, color='darkorange')
            plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
            plt.xlim([-0.00099, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Roc Curve-multi-exit')
            self.trainer.logger.experiment.add_figure(f'ROC-multi-exit-{candidate}', fig)
            self.log(f'ROC-AUC-Score-multi-exit-{candidate}',auc(fpr,tpr))
            df = pd.DataFrame()
            df['Threshold'] = thresholds
            df['FPR'] = np.array(fpr)
            df['TPR'] = np.array(tpr)
            df['acc'] = np.array(acc_list)
            pathlib.Path(f'output/{self.enc_type}/{self.mode}/threshold_candidate_{candidate}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}').mkdir(parents=True, exist_ok=True)
            df.to_csv(f'output/{self.enc_type}/{self.mode}/threshold_candidate_{candidate}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}/FPR_TPR-multi-exit.csv')
    def plot_roc_curve_final_exit(self,all_data,labels,all_sift_distances):
        all_data = [torch.tensor(exit_data,device=self.device) for exit_data in all_data ]
        value_list = []
        pred_labels_list = []
        second_values_list = []
        second_labels_list = []
        confidence_score_list = []
        for exit in range(0, 5):
            values,pred_labels,second_values,second_labels,confidence_score  = self.get_data_top_val(all_data[exit],all_sift_distances,labels,exit)#这里会减掉最大的logit所以ood的数据负的更多
            value_list.append(values)
            pred_labels_list.append(pred_labels)
            second_values_list.append(second_values)
            second_labels_list.append(second_labels)
            confidence_score_list.append(confidence_score)
        value_multi_exit = torch.stack(value_list).transpose(0,1)
        pred_labels_multi_exit = torch.stack(pred_labels_list).transpose(0,1)
        second_values_multi_exit = torch.stack(second_values_list).transpose(0,1)
        second_labels_multi_exit = torch.stack(second_labels_list).transpose(0,1)
        confidence_score_multi_exit = torch.stack(confidence_score_list).transpose(0,1)

        # 在所有confidence都不小于candidate情况下，把confidence最小的值都赋值给最后一个出口，有可能是他自己
        no_threshold_constrain_where_to_output = torch.min(confidence_score_multi_exit, dim=1)[1]
        no_threshold_constrain_where_to_output_confidence = torch.min(confidence_score_multi_exit, dim=1)[0]
        no_threshold_constrain_where_to_output_values = value_multi_exit[torch.arange(confidence_score.shape[0]), no_threshold_constrain_where_to_output]
        no_threshold_constrain_where_to_output_pred_labels = pred_labels_multi_exit[torch.arange(confidence_score.shape[0]), no_threshold_constrain_where_to_output]

        value_multi_exit[:,4] = no_threshold_constrain_where_to_output_values
        pred_labels_multi_exit[:,4] = no_threshold_constrain_where_to_output_pred_labels


        # output_threshold_candidate = [0.05,0.1,0.15,0.2]
        # for candidate in output_threshold_candidate:
        #     temp_confidence = confidence_score_multi_exit
        #     temp_confidence[:][4]=-1
        #     threshold_constrain_where_to_output = temp_confidence < candidate
        #     where_to_output = torch.argmax(threshold_constrain_where_to_output.ne(0)*1.0,dim=1)
        #     where_to_output_counts = torch.bincount(where_to_output, minlength=5)
        #     print(f'candidate:{candidate},{where_to_output_counts}')
        #     where_to_output_values = torch.gather(value_multi_exit, 1, where_to_output.unsqueeze(1)).squeeze()
        #     where_to_output_labels = torch.gather(pred_labels_multi_exit, 1, where_to_output.unsqueeze(1)).squeeze()
        candidate = 'final'
        values = no_threshold_constrain_where_to_output_values.cpu().numpy()
        pred_labels = no_threshold_constrain_where_to_output_pred_labels.cpu().numpy()
        self.log_test_statistics(values,pred_labels,labels,exit)
        thresholds = np.flip(np.geomspace(np.max(values),np.min(values),num=100000))#geomspace 取对数后构成等差数列
        fpr = []
        tpr = []
        acc_list = []
        aurocNew = 0.0
        fprTemp = 1.0
        fp=0
        tp=0
        tp_cor=0
        fpr_at_095 = 0
        tpr_at_095 = 0


        for t in tqdm(range(len(thresholds))):
            fp,tp,tp_cor,acc = self.get_fpr_tpr(values,pred_labels,labels,thresholds[t])#tp_cor
            # both fp and tp is decreascing
            if tp >= 0.95:
                fpr_at_095 = fp
                tpr_at_095 = tp
                acc_at_095 = acc
            if fp >= 1-0.9: # tn <= 0.9
                fpr_at_tn_090 = fp
                tpr_at_tn_090 = tp
                acc_at_tn_090 = acc

            fpr.append(fp)
            tpr.append(tp)
            acc_list.append(acc)
            aurocNew += (-fp+fprTemp)*tp_cor
            fprTemp = fp
        aurocNew += fp * tp_cor
        self.log(f'CAUROC-multi-exit-{candidate}',aurocNew)
        self.log(f"fpr-at-tpr-095-multi-exit-{candidate}",fpr_at_095)
        self.log(f"tpr-at-tpr-095-multi-exit-{candidate}",tpr_at_095)
        self.log(f"tnr-at-tpr-095-multi-exit-{candidate}",1-fpr_at_095)
        self.log(f"acc-at-tpr-095-multi-exit-{candidate}",acc_at_095)

        self.log(f"fpr-at-tnr-090-multi-exit-{candidate}",fpr_at_tn_090)
        self.log(f"tnr-at-tnr-090-multi-exit-{candidate}",1-fpr_at_tn_090)
        self.log(f"tpr-at-tnr-090-multi-exit-{candidate}",tpr_at_tn_090)
        self.log(f"acc-at-tnr-090-multi-exit-{candidate}",acc_at_tn_090)
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
        plt.xlim([-0.00099, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Roc Curve-multi-exit')
        self.trainer.logger.experiment.add_figure(f'ROC-multi-exit-{candidate}', fig)
        self.log(f'ROC-AUC-Score-multi-exit-{candidate}',auc(fpr,tpr))
        df = pd.DataFrame()
        df['Threshold'] = thresholds
        df['FPR'] = np.array(fpr)
        df['TPR'] = np.array(tpr)
        df['acc'] = np.array(acc_list)
        pathlib.Path(f'output/{self.enc_type}/{self.mode}/threshold_candidate_{candidate}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}').mkdir(parents=True, exist_ok=True)
        df.to_csv(f'output/{self.enc_type}/{self.mode}/threshold_candidate_{candidate}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}/FPR_TPR-multi-exit.csv')

    # def plot_roc_curve_final_exit(self,all_data,labels,all_sift_distances):
    #     all_data = torch.tensor(all_data,device=self.device)
    #     value_list = []
    #     for exit in range(0, 5):
    #         values,pred_labels,second_values,second_labels,confidence_score  = self.get_data_top_val(all_data,all_sift_distances,labels,exit)#这里会减掉最大的logit所以ood的数据负的更多
    #         value_list.append(values)
    #
    #     ###在这一步之前就应该拿到test数据
    #     values = values.cpu().numpy()
    #     pred_labels = pred_labels.cpu().numpy()
    #     self.log_test_statistics(values,pred_labels,labels,exit)
    #     thresholds = np.flip(np.geomspace(np.max(values),np.min(values),num=100000))#geomspace 取对数后构成等差数列
    #     fpr = []
    #     tpr = []
    #     acc_list = []
    #     aurocNew = 0.0
    #     fprTemp = 1.0
    #     fp=0
    #     tp=0
    #     tp_cor=0
    #     fpr_at_095 = 0
    #     tpr_at_095 = 0
    #
    #
    #     for t in tqdm(range(len(thresholds))):
    #         fp,tp,tp_cor,acc = self.get_fpr_tpr(values,pred_labels,labels,thresholds[t])#tp_cor
    #         # both fp and tp is decreascing
    #         if tp >= 0.95:
    #             fpr_at_095 = fp
    #             tpr_at_095 = tp
    #             acc_at_095 = acc
    #         if fp >= 1-0.9: # tn <= 0.9
    #             fpr_at_tn_090 = fp
    #             tpr_at_tn_090 = tp
    #             acc_at_tn_090 = acc
    #
    #         fpr.append(fp)
    #         tpr.append(tp)
    #         acc_list.append(acc)
    #         aurocNew += (-fp+fprTemp)*tp_cor
    #         fprTemp = fp
    #     aurocNew += fp * tp_cor
    #     self.log(f'CAUROC-multi-exit',aurocNew)
    #     self.log(f"fpr-at-tpr-095-multi-exit",fpr_at_095)
    #     self.log(f"tpr-at-tpr-095-multi-exit",tpr_at_095)
    #     self.log(f"tnr-at-tpr-095-multi-exit",1-fpr_at_095)
    #     self.log(f"acc-at-tpr-095-multi-exit",acc_at_095)
    #
    #     self.log(f"fpr-at-tnr-090-multi-exit",fpr_at_tn_090)
    #     self.log(f"tnr-at-tnr-090-multi-exit",1-fpr_at_tn_090)
    #     self.log(f"tpr-at-tnr-090-multi-exit",tpr_at_tn_090)
    #     self.log(f"acc-at-tnr-090-multi-exit",acc_at_tn_090)
    #     fig = plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange')
    #     plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
    #     plt.xlim([-0.00099, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'Roc Curve-multi-exit')
    #     # self.trainer.logger.experiment.log_image('ROC', fig, description=f'ROC TEST')
    #     self.trainer.logger.experiment.add_figure(f'ROC-multi-exit', fig)
    #     self.log(f'ROC-AUC-Score-multi-exit',auc(fpr,tpr))
    #     df = pd.DataFrame()
    #     df['Threshold'] = thresholds
    #     df['FPR'] = np.array(fpr)
    #     df['TPR'] = np.array(tpr)
    #     df['acc'] = np.array(acc_list)
    #     # log_table('FPR_TPR', df,experiment=self.trainer.logger.experiment) #这个也没法改
        # df.to_csv(f'output/{self.enc_type}/{self.mode}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}/FPR_TPR-multi-exit.csv')
    def test_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        pathlib.Path(f'output/{self.enc_type}/{self.mode}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}').mkdir(parents=True, exist_ok=True)
        all_sift = torch.vstack([x[3].reshape(x[3].shape[0], -1) for x in outputs])
        all_sift_hat = torch.vstack([x[2].reshape(x[2].shape[0], -1) for x in outputs])
        all_sift_distances = ((all_sift - all_sift_hat) ** 2).mean(axis=1)
        all_sift_distances /= all_sift_distances.max()
        all_sift_distances = all_sift_distances

        if self.is_tested == 0:
            self.calculate_gaussian()
        elif self.is_tested == 1:
            self.class_gaussians = self.early_gaussians
        if self.mode == 'test_all_case':
            for exit in range(0, 5):
                all_data = torch.vstack([x[6][exit] for x in outputs]).cpu().numpy()
                all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()  # true label
                all_known_labels = all_labels[all_labels < self.class_num]
                all_known_data = all_data[all_labels < self.class_num]

                if self.log_tsne:
                    x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_known_data)
                    x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])
                    x_te_proj_df['label'] = all_known_labels
                    fig = plt.figure()
                    ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                                         palette='tab20',
                                         hue='label',
                                         linewidth=0,
                                         alpha=0.6,
                                         s=7)
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    if self.is_tested == 0:
                        self.trainer.logger.experiment.add_figure(f'Train T-SNE_exit_{exit}, Final TEST', fig)
                    elif self.is_tested == 1:
                        self.trainer.logger.experiment.add_figure(f'Test T-SNE Without Unknown_exit_{exit}', fig)
                if self.is_tested == 0:
                    self.is_tested = 1
                    return
                elif self.is_tested == 1:
                    if self.log_tsne:
                        print("start tsne")
                        x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_data)
                        x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])
                        x_te_proj_df['label'] = all_labels
                        fig = plt.figure()
                        ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                                             palette='tab20',
                                             hue='label',
                                             linewidth=0,
                                             alpha=0.6,
                                             s=7)
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        # self.trainer.logger.experiment.log_image('T-SNE With Unknown', fig, description=f'Final TEST')
                        self.trainer.logger.experiment.add_figure(f'T-SNE With Unknown_exit_{exit}', fig)
                        print("end tsne")


                    self.plot_roc_curve_each_exit(all_data, all_labels, all_sift_distances, exit)  # all_data是l_feature

            all_data = [torch.vstack([x[6][exit] for x in outputs]).cpu().numpy() for exit in range(0, 5)]
            all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()  # true label

            self.plot_roc_curve_multi_exit(all_data, all_labels, all_sift_distances)
            self.plot_roc_curve_final_exit(all_data, all_labels, all_sift_distances)

        else:
            for exit in range(0,5):
                all_data = torch.vstack([x[6][exit] for x in outputs]).cpu().numpy()
                all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()  # true label
                all_known_labels = all_labels[all_labels < self.class_num]
                all_known_data = all_data[all_labels < self.class_num]

                if self.log_tsne:
                    x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_known_data)
                    x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])
                    x_te_proj_df['label'] = all_known_labels
                    fig = plt.figure()
                    ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                                         palette='tab20',
                                         hue='label',
                                         linewidth=0,
                                         alpha=0.6,
                                         s=7)
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    if self.is_tested == 0:
                        self.trainer.logger.experiment.add_figure(f'Train T-SNE_exit_{exit}, Final TEST', fig)
                    elif self.is_tested == 1:
                        self.trainer.logger.experiment.add_figure(f'Test T-SNE Without Unknown_exit_{exit}', fig)
                if self.is_tested == 0:
                    self.is_tested = 1
                    return
                elif self.is_tested == 1:
                    if self.log_tsne:
                        print("start tsne")
                        x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_data)
                        x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])
                        x_te_proj_df['label'] = all_labels
                        fig = plt.figure()
                        ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                                             palette='tab20',
                                             hue='label',
                                             linewidth=0,
                                             alpha=0.6,
                                             s=7)
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                        # self.trainer.logger.experiment.log_image('T-SNE With Unknown', fig, description=f'Final TEST')
                        self.trainer.logger.experiment.add_figure(f'T-SNE With Unknown_exit_{exit}', fig)
                        print("end tsne")

                    if self.mode == 'test_each_exit':
                        self.plot_roc_curve_each_exit(all_data, all_labels, all_sift_distances,exit)  # all_data是l_feature

            all_data = [torch.vstack([x[6][exit] for x in outputs]).cpu().numpy() for exit in range(0,5)]
            all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()  # true label

            if self.mode == 'test_multi_exit':
                self.plot_roc_curve_multi_exit(all_data, all_labels, all_sift_distances)
            elif self.mode == 'test_final_exit':
                self.plot_roc_curve_final_exit(all_data, all_labels, all_sift_distances)



    def validation_epoch_end(self, validation_step_outputs):
        return None
        all_data = torch.vstack([x[0] for x in validation_step_outputs]).cpu().numpy()
        all_labels = torch.hstack([x[1] for x in validation_step_outputs]).cpu().numpy()
        neg_samples_count = float(np.sum(all_labels == self.class_num)) #实际的ood数据数量
        pos_samples_count = float(np.sum(all_labels < self.class_num))  #实际的id数据数量
        if self.log_tsne:
            x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_data)

            x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])

            x_te_proj_df['label'] = all_labels
            fig = plt.figure()
            ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df,
                    palette='tab20',
                    hue='label',
                    linewidth=0,
                    alpha=0.6,
                    s=7)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # self.trainer.logger.experiment.add_figure('pca', fig, global_step=self.trainer.current_epoch)
            # self.trainer.logger.experiment.log_image('T-SNE', fig, description=f'epoch: {self.trainer.current_epoch}')
            self.trainer.logger.experiment.add_figure('T-SNE_epoch_{}'.format(self.trainer.current_epoch), fig)
        if self.trainer.current_epoch >= 1:
            all_data = torch.tensor(all_data,device=self.device)        #输出的特征图
            all_labels = torch.tensor(all_labels,device=self.device)    #真实标签
            labels = self.classify_data(all_data)                       #预测结果
            true_labels = labels == all_labels                          #预测结果==真实标签True/False
            f1score = f1_score(all_labels.detach().cpu().numpy(),labels.detach().cpu().numpy(),average='macro')
            self.log(f"f1_score", f1score )
            pos_labels_exact_predicted = float(torch.sum(true_labels[all_labels < self.class_num]))     #预测正确的数据里面id数据的数量 包括id对齐（False在sum的时候是0）
            # 上面这个只用来计算分类器的分类结果

            # 下面统计binary classification的结果

            n = float(torch.sum(all_labels == self.class_num))     # 实际的ood数据数量
            p = float(torch.sum(all_labels < self.class_num))      # 实际的id数据数量
            pn = float(torch.sum(labels == self.class_num))        # 预测的ood数据数量
            pp = float(torch.sum(labels < self.class_num))        # 预测的id数据数量

            tn = float(torch.sum(all_labels[labels == self.class_num] == self.class_num))
            fn = float(torch.sum(all_labels[labels == self.class_num] < self.class_num))
            tp = float(torch.sum(all_labels[labels < self.class_num] < self.class_num))
            fp = float(torch.sum(all_labels[labels < self.class_num] == self.class_num))

            exact_acc_pos = pos_labels_exact_predicted/p
            self.log(f"val_false_negative",fn/p)
            self.log(f"val_false_postive",fp/n)
            self.log(f"val_acc_pos",tp/p)
            self.log(f"val_acc_neg",tn/n)
            self.log(f"exact_acc_pos",exact_acc_pos)
            print("Acc:",exact_acc_pos,"F1",f1score)
        else:
            self.log(f"f1_score", 0)

    def initialize_gaussians(self):
        self.class_counts = torch.zeros(self.class_num, device=self.device)
        self.layer_size = 0
        if self.enc_type == 'resnet18_semantic_temp_early_exit':
            self.layer_size_0 = 64 + self.latent_dim
            self.layer_size_1 = 64 + 64 + self.latent_dim
            self.layer_size_2 = 64 + 64 + 512 + self.latent_dim
            self.layer_size_3 = 64 + 64 + 512 + 512 + self.latent_dim
            self.layer_size_4 = 64 + 64 + 512 + 512 + 512 + self.latent_dim
        else:
            raise Exception("Please check enc_type")

        self.early_temp_means_0 = torch.zeros((self.class_num,self.layer_size_0),device=self.device)
        self.early_temp_cov_0 = torch.zeros(((self.class_num,self.layer_size_0,self.layer_size_0)),device=self.device)
        self.early_means_0 = torch.zeros((self.class_num,self.layer_size_0),device=self.device)
        self.early_cov_0 = torch.zeros(((self.class_num,self.layer_size_0,self.layer_size_0)),device=self.device)



        self.early_temp_means_1 = torch.zeros((self.class_num, self.layer_size_1), device=self.device)
        self.early_temp_cov_1 = torch.zeros(((self.class_num, self.layer_size_1, self.layer_size_1)),
                                            device=self.device)
        self.early_means_1 = torch.zeros((self.class_num, self.layer_size_1), device=self.device)
        self.early_cov_1 = torch.zeros(((self.class_num, self.layer_size_1, self.layer_size_1)), device=self.device)



        self.early_temp_means_2 = torch.zeros((self.class_num, self.layer_size_2), device=self.device)
        self.early_temp_cov_2 = torch.zeros(((self.class_num, self.layer_size_2, self.layer_size_2)),
                                            device=self.device)
        self.early_means_2 = torch.zeros((self.class_num, self.layer_size_2), device=self.device)
        self.early_cov_2 = torch.zeros(((self.class_num, self.layer_size_2, self.layer_size_2)), device=self.device)



        self.early_temp_means_3 = torch.zeros((self.class_num, self.layer_size_3), device=self.device)
        self.early_temp_cov_3 = torch.zeros(((self.class_num, self.layer_size_3, self.layer_size_3)),
                                            device=self.device)
        self.early_means_3 = torch.zeros((self.class_num, self.layer_size_3), device=self.device)
        self.early_cov_3 = torch.zeros(((self.class_num, self.layer_size_3, self.layer_size_3)), device=self.device)


        self.early_temp_means_4 = torch.zeros((self.class_num, self.layer_size_4), device=self.device)
        self.early_temp_cov_4 = torch.zeros(((self.class_num, self.layer_size_4, self.layer_size_4)),
                                            device=self.device)
        self.early_means_4 = torch.zeros((self.class_num, self.layer_size_4), device=self.device)
        self.early_cov_4 = torch.zeros(((self.class_num, self.layer_size_4, self.layer_size_4)), device=self.device)


    def configure_optimizers(self):

        params = self.parameters()
        lr=self.lr

        if self.opt == 'sgd':
            print('SGD')
            optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9, dampening=0, weight_decay=1e-4,nesterov=True)
        if self.opt == 'adam':
            print('ADAM')
            optimizer = torch.optim.Adam(params, lr=lr,weight_decay=1e-4)
        lr_scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.step_gamma)
        # scheduler = CyclicLR(optimizer, base_lr=self.step_gamma, max_lr=self.lr)
        # lr_scheduler = {
        #     'scheduler': scheduler,
        #     'interval': 'step', # or 'epoch'
        #     'frequency': 1
        # }
        # optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        if self.trainer.current_epoch < 20:
            all_classes = self.class_num
            self.class_counts = torch.zeros(all_classes,device=self.device)
            self.class_temp_means = torch.zeros((all_classes,self.latent_dim),device=self.device)
            self.class_temp_cov = torch.zeros(((all_classes,self.latent_dim,self.latent_dim)),device=self.device)
            self.class_means = torch.zeros((all_classes,self.latent_dim),device=self.device)
            self.class_cov = torch.zeros(((all_classes,self.latent_dim,self.latent_dim)),device=self.device)
            self.class_bounds = torch.zeros(all_classes,device=self.device)
            self.class_counts.requires_grad = False
            self.class_temp_cov.requires_grad = False
            self.class_cov.requires_grad = False
            self.class_temp_means.requires_grad = False
            self.class_means.requires_grad = False
            self.precentiles = [[] for i in range(all_classes)]
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str,default=None)
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim", type=int, default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--generation_epoch", type=int,default=20)
        parser.add_argument("--class_num", type=int, default=10)
        parser.add_argument("--lower_bound", type=float, default = 0.05)
        parser.add_argument("--step_size", type=int, default=100)
        parser.add_argument("--step_gamma", type=float, default = 0.1)
        parser.add_argument("--cov_scaling",type=float, default = 5.0)
        parser.add_argument("--log_tsne",action='store_true')
        parser.add_argument("--recon_weight",type=float, default = 0.1)
        parser.add_argument("--gen_weight",type=float,default = 0.5)
        parser.add_argument("--weights", default=None, type=str)
        parser.add_argument("--finetune",action='store_true')
        parser.add_argument("--finetune_lr",type=float, default = 0.0001)
        parser.add_argument("--opt", type=str, default="sgd")
        parser.add_argument("--ae_features",action='store_true')
        parser.add_argument("--backbone_checkpoint",type=str,default='checkpoints/resnet18_semantic_temp/cifar_known/version_2/checkpoint-epoch=475-f1_score=0.789.ckpt')
        parser.add_argument("--backbone_type",type=str,default='resnet18_semantic_temp')
        parser.add_argument("--exit_num",type=int,default=None)
        parser.add_argument("--mode",type=str,default=None)
        return parser



    def calculate_gaussian(self):
            '''
            This is used in testing, to run a training epoch (with no gradients) so we can calculate the gaussians.
            '''

            early_gaussians =[[],[],[],[],[]]

            # calculate gaussian param
            for i in range(self.class_num):
                self.early_means_0[i,:] = self.early_temp_means_0[i,:]/self.class_counts[i]
                self.early_cov_0[i,:,:] = self.early_temp_cov_0[i,:,:]/self.class_counts[i] - torch.matmul(self.early_means_0[i,:].view(1,-1).T,self.early_means_0[i,:].view(1,-1))
                self.early_cov_0[i,:,:] = (self.early_cov_0[i,:,:] + torch.t(self.early_cov_0[i,:,:]))/2

                self.early_means_1[i, :] = self.early_temp_means_1[i, :] / self.class_counts[i]
                self.early_cov_1[i, :, :] = self.early_temp_cov_1[i, :, :] / self.class_counts[i] - torch.matmul(self.early_means_1[i, :].view(1, -1).T, self.early_means_1[i, :].view(1, -1))
                self.early_cov_1[i, :, :] = (self.early_cov_1[i, :, :] + torch.t(self.early_cov_1[i, :, :])) / 2

                self.early_means_2[i, :] = self.early_temp_means_2[i, :] / self.class_counts[i]
                self.early_cov_2[i, :, :] = self.early_temp_cov_2[i, :, :] / self.class_counts[i] - torch.matmul(self.early_means_2[i, :].view(1, -1).T, self.early_means_2[i, :].view(1, -1))
                self.early_cov_2[i, :, :] = (self.early_cov_2[i, :, :] + torch.t(self.early_cov_2[i, :, :])) / 2

                self.early_means_3[i, :] = self.early_temp_means_3[i, :] / self.class_counts[i]
                self.early_cov_3[i, :, :] = self.early_temp_cov_3[i, :, :] / self.class_counts[i] - torch.matmul(self.early_means_3[i, :].view(1, -1).T, self.early_means_3[i, :].view(1, -1))
                self.early_cov_3[i, :, :] = (self.early_cov_3[i, :, :] + torch.t(self.early_cov_3[i, :, :])) / 2

                self.early_means_4[i, :] = self.early_temp_means_4[i, :] / self.class_counts[i]
                self.early_cov_4[i, :, :] = self.early_temp_cov_4[i, :, :] / self.class_counts[i] - torch.matmul(self.early_means_4[i, :].view(1, -1).T, self.early_means_4[i, :].view(1, -1))
                self.early_cov_4[i, :, :] = (self.early_cov_4[i, :, :] + torch.t(self.early_cov_4[i, :, :])) / 2

            # \Sigma+\lambda I for regularization

            # exit 0
            increases = 1000
            for i in range(self.class_num):
                class_increase = increases
                cov_copy = torch.clone(self.early_cov_0[i, :, :])
                cov_copy += torch.eye(self.early_cov_0[i, :, :].shape[0], device=self.device) * 0.000001 * increases
                while True:
                    try:
                        # torch.cholesky(cov_copy)
                        torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_0[i, :], cov_copy)
                        break
                    except:
                        cov_copy += torch.eye(self.early_cov_0[i, :, :].shape[0], device=self.device) * 0.000001
                        class_increase += 1
                if class_increase > increases:
                    increases = class_increase
            increases += 2
            print("Increases:", increases)
            for i in range(self.class_num):
                self.early_cov_0[i, :, :] += torch.eye(self.early_cov_0[i, :, :].shape[0],
                                                       device=self.device) * 0.000001 * increases
                early_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_0[i, :],
                                                                                        self.early_cov_0[i, :, :])
                early_gaussians[0].append(early_dist)

            # exit 1
            increases = 2000
            for i in range(self.class_num):
                class_increase = increases
                cov_copy = torch.clone(self.early_cov_1[i,:,:])
                cov_copy += torch.eye(self.early_cov_1[i,:,:].shape[0],device=self.device)*0.000001*increases
                while True:
                    try:
                        # torch.cholesky(cov_copy)
                        torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_1[i,:],cov_copy)
                        break
                    except:
                        cov_copy += torch.eye(self.early_cov_1[i,:,:].shape[0],device=self.device)*0.000001
                        class_increase += 1
                if class_increase > increases:
                    increases = class_increase
            increases+=2
            print("Increases:", increases)
            for i in range(self.class_num):
                self.early_cov_1[i,:,:] += torch.eye(self.early_cov_1[i,:,:].shape[0],device=self.device)*0.000001*increases
                early_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_1[i,:],self.early_cov_1[i,:,:])
                early_gaussians[1].append(early_dist)


            # exit 2
            increases = 3000
            for i in range(self.class_num):
                class_increase = increases
                cov_copy = torch.clone(self.early_cov_2[i,:,:])
                cov_copy += torch.eye(self.early_cov_2[i,:,:].shape[0],device=self.device)*0.000001*increases
                while True:
                    try:
                        # torch.cholesky(cov_copy)
                        torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_2[i,:],cov_copy)
                        break
                    except:
                        cov_copy += torch.eye(self.early_cov_2[i,:,:].shape[0],device=self.device)*0.000001
                        class_increase += 1
                if class_increase > increases:
                    increases = class_increase
            increases+=2
            print("Increases:", increases)
            for i in range(self.class_num):
                self.early_cov_2[i,:,:] += torch.eye(self.early_cov_2[i,:,:].shape[0],device=self.device)*0.000001*increases
                early_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_2[i,:],self.early_cov_2[i,:,:])
                early_gaussians[2].append(early_dist)

            # exit 3
            increases = 4000
            for i in range(self.class_num):
                class_increase = increases
                cov_copy = torch.clone(self.early_cov_3[i,:,:])
                cov_copy += torch.eye(self.early_cov_3[i,:,:].shape[0],device=self.device)*0.000001*increases
                while True:
                    try:
                        # torch.cholesky(cov_copy)
                        torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_3[i,:],cov_copy)
                        break
                    except:
                        cov_copy += torch.eye(self.early_cov_3[i,:,:].shape[0],device=self.device)*0.000001
                        class_increase += 1
                if class_increase > increases:
                    increases = class_increase
            increases+=2
            print("Increases:", increases)
            for i in range(self.class_num):
                self.early_cov_3[i,:,:] += torch.eye(self.early_cov_3[i,:,:].shape[0],device=self.device)*0.000001*increases
                early_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_3[i,:],self.early_cov_3[i,:,:])
                early_gaussians[3].append(early_dist)

            # exit 4
            increases = 5000
            for i in range(self.class_num):
                class_increase = increases
                cov_copy = torch.clone(self.early_cov_4[i,:,:])
                cov_copy += torch.eye(self.early_cov_4[i,:,:].shape[0],device=self.device)*0.000001*increases
                while True:
                    try:
                        # torch.cholesky(cov_copy)
                        torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_4[i,:],cov_copy)
                        break
                    except:
                        cov_copy += torch.eye(self.early_cov_4[i,:,:].shape[0],device=self.device)*0.000001
                        class_increase += 1
                if class_increase > increases:
                    increases = class_increase
            increases+=2
            print("Increases:", increases)
            for i in range(self.class_num):
                self.early_cov_4[i,:,:] += torch.eye(self.early_cov_4[i,:,:].shape[0],device=self.device)*0.000001*increases
                early_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.early_means_4[i,:],self.early_cov_4[i,:,:])
                early_gaussians[4].append(early_dist)
            self.unfreeze()
            self.early_gaussians = early_gaussians
