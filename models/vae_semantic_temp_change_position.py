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
from models.resnet_encoders_semantic_temp import resnet18_encoder_semantic_change_position
import pathlib
from time import time
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

class GMM_VAE_Contrastive(pl.LightningModule):
    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
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
        transmit_dim = None,
        **kwargs
    ):

        super(GMM_VAE_Contrastive, self).__init__()

        self.save_hyperparameters()
        self.known_dataset = known_dataset
        self.unknown_dataset = unknown_dataset
        self.lr = lr
        self.step_size = step_size
        self.step_gamma = step_gamma
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.class_num = class_num
        self.generation_epoch = generation_epoch
        self.cov_scaling = cov_scaling
        self.log_tsne = log_tsne
        self.is_tested = 0
        self.criterion = nn.CrossEntropyLoss()
        self.gen_weight = gen_weight
        self.weights = weights
        self.finetune = finetune
        self.finetune_lr = finetune_lr
        self.opt = opt
        self.ae_features = ae_features
        self.margin_max_distance = margin_max_distance
        self.sample_count = sample_count
        self.noise = noise
        self.transmit_dim = transmit_dim
        valid_encoders = {
            'resnet18_semantic_temp_change_position':{'enc':resnet18_encoder_semantic_change_position,'dec':None},
        }

        self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1,self.transmit_dim)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.decoder_fc = nn.Linear(self.latent_dim,self.class_num)
        self.metric_loss = TripletMarginLossModded(margin=0.1,neg_margin=0.15)
        self.precentiles = []
        self.class_gaussians = []
        self.lower_bound = lower_bound
        self.class_gaussian = []

        if self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.enc_type = enc_type
        self.ae = None
    def initilize_ae(self):
        self.ae = ae_dict['cifar_known'][0]()
        self.ae = self.ae.load_from_checkpoint(ae_dict['cifar_known'][1])
        self.ae.cuda()
        self.ae.freeze()
    def forward(self, x):
        x = self.encoder(x,self.noise)
        # return x
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z
    def final_distribution_logit(self,x,noise):
        final_feature,logit,early_features, _ = self.encoder.get_layer_output_for_multi_exit(x,noise)
        final_mu = self.fc_mu(final_feature)
        final_var = self.fc_var(final_feature)
        return final_mu,final_var,logit, early_features
    def _run_step(self, x):
        x,logit,[dec_mu,dec_log_var] = self.encoder(x,self.noise)
        # return x
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        mu = torch.hstack([mu,dec_mu])
        log_var = torch.hstack([log_var,dec_log_var])
        p, q, z = self.sample(mu, log_var)
        # return z,q
        return z, mu, log_var, q,logit

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    # 浓缩，只剩下每个类别都不属于的数据
    def filter_samples(self,all_samples,labels):
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(all_samples) < self.class_bounds[i]
            all_samples = all_samples[probs]
            labels = labels[probs]
        return all_samples,labels

    def filter_just_samples(self,samples):
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(samples) < self.class_bounds[i]
            samples = samples[probs]
        return samples




    def step(self, batch, batch_idx):
        x, y = batch
        cur_classes = torch.unique(y).long()

        z ,mu,log_var, q,logit= self._run_step(x)

        kl = 0
        log_qz = q.log_prob(z)
        for i in cur_classes:

            y_count = (y==i).sum()
            rel_zs = z[y==i,:]
            self.class_temp_means[i,:] += rel_zs.sum(axis = 0).detach()
            self.class_temp_cov[i,:,:] += torch.matmul(rel_zs.T,rel_zs).detach()
            self.class_counts[i] += y_count.detach()
            #Estimation:
            cp = torch.distributions.Normal(self.class_means[i,:].repeat(y_count,1), torch.ones_like(rel_zs)*(0.1 if i < self.class_num else 1))
            log_pz = cp.log_prob(rel_zs)
            kl_class = log_qz[y==i] - log_pz
            kl_class = kl_class.mean()
            kl_class *= self.kl_coeff
            kl += kl_class / len(cur_classes)
            #Analytical
            # rel_mu = mu[y==i,:]
            # rel_var = torch.exp(log_var[y==i,:]/2)
            # q = torch.distributions.Normal(rel_mu, rel_var)
            # p = torch.distributions.Normal(self.class_means[i,:].repeat(y_count,1), torch.ones_like(rel_zs)*(0.1 if i < self.class_num else 1))
            # kl_loss = torch.distributions.kl.kl_divergence(q,p).mean()*self.kl_coeff
            # kl += kl_loss / len(cur_classes)
            if self.trainer.current_epoch > 0:
                class_ll = self.class_gaussian[i].log_prob(rel_zs).detach()
                self.precentiles[i].append(class_ll)

        contrastive_loss = self.metric_loss(z,y)
        ce_loss = self.criterion(logit,y)
        if self.trainer.current_epoch < self.generation_epoch:
            kl = 0
        loss = ce_loss + contrastive_loss + kl
        logs = {
            "ce_loss":ce_loss,
            "loss": loss,
            "kl": kl,
            "triplet_loss": contrastive_loss,
        }
        return loss, logs



    def on_test_epoch_start(self) -> None:
        self.initilize_ae()
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.enc_type == 'resnet18_semantic_temp_change_position':
            mu= self.fc_mu(self.encoder(x, self.noise)[0])
            logit = self.encoder(x,self.noise)[1]
            decx_mu = self.encoder(x,self.noise)[2][0]
        z = torch.hstack([mu,decx_mu])
        l_features = [torch.flatten(self.encoder.get_layer_output(x,self.noise,i),1) for i in range(0,5)]  + [z]
        # l_features = [decx_mu]
        # l_features = [mu]
        l_features = torch.hstack(l_features)
        # l_features = torch.hstack([l_features,mu])
        # z ,x_hat,p, q= self._run_step(x_hat)
        # if self.is_tested == 1:
            # z2 ,x_hat2 ,p, q= self._run_step(x_hat)
            # x_hat[y==self.class_num] = x_hat2[y==self.class_num]
            # z[y==self.class_num] = z2[y==self.class_num]
        cur_classes = torch.unique(y)
        if self.trainer.testing and self.is_tested == 0:
            for i in cur_classes:
                y_count = (y==i).sum()
                rel_zs = z[y==i,:]
                rel_l_features = l_features[y==i,:]
                self.class_temp_means[i,:] += rel_zs.sum(axis = 0).detach()
                self.class_temp_cov[i,:,:] += torch.matmul(rel_zs.T,rel_zs).detach()
                self.class_counts[i] += y_count.detach()
                self.early_temp_means[i,:] = rel_l_features.sum(axis = 0).detach()
                self.early_temp_cov[i,:,:] = torch.matmul(rel_l_features.T,rel_l_features).detach()

        # x_hat = x + (0.01)*torch.randn(x.shape,device=self.device)
        # x_hat = self.rand_transforms(x)
        x_hat=x
        if self.is_tested != 0:
            xshape = x.shape[1]
            chunk_size = int(x.shape[0] / 4)
            x_hat = []
            for i in range(4):
                x_hat.append(self.ae(x[chunk_size * i:chunk_size * (i + 1)]))   #防止out of memory
            x_hat = torch.vstack(x_hat)
            desc_x_encoder = torch.flatten(self.encoder.get_layer_output(x, self.noise,1), 1)
            desc_x_hat_encoder = torch.flatten(self.encoder.get_layer_output(x_hat,self.noise, 1), 1)
            desc_x = desc_x_encoder
            desc_x_hat = desc_x_hat_encoder
        else:
            desc_x = 0
            desc_x_hat = 0
            x_hat = x
        if self.enc_type == 'resnet18_semantic_temp_change_position':
            return z,y,x_hat,x,desc_x,desc_x_hat,l_features,logit
        elif self.enc_type == 'resnet18_semantic_1':
            return z,y,x_hat,x,desc_x,desc_x_hat,l_features
    def training_epoch_end(self, outputs):
        class_gaussians = []
        bad_classes = 0
        for i in range(self.class_num):
            if self.trainer.current_epoch > 0:
                self.precentiles[i] = torch.hstack(self.precentiles[i])
                self.class_bounds[i] = torch.quantile(self.precentiles[i],self.lower_bound)
            self.class_means[i,:] = self.class_temp_means[i,:]/self.class_counts[i]
            self.class_cov[i,:,:] = self.class_temp_cov[i,:,:]/self.class_counts[i] - torch.matmul(self.class_means[i,:].view(1,-1).T,self.class_means[i,:].view(1,-1))
            try:
                dist = torch.distributions.multivariate_normal.MultivariateNormal(self.class_means[i,:],self.class_cov[i,:,:])
            except:
                bad_classes += 1
                dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros_like(self.class_means[i,:],device=self.device),torch.eye(self.class_cov[i,:,:].shape[0],device=self.device))
            class_gaussians.append(dist)

        cur_num = self.class_num
        self.full_gaussian_mean = self.class_means.sum(axis=0) / self.class_counts.sum()
        self.full_gaussian_cov = self.class_temp_cov.sum(axis=0) / self.class_counts.sum() - torch.matmul(self.full_gaussian_mean.view(1,-1).T,self.full_gaussian_mean.view(1,-1))
        self.class_gaussian = class_gaussians
        self.precentiles = [[] for i in range(cur_num)]
        self.class_counts = torch.zeros(cur_num,device=self.device)
        self.class_temp_means = torch.zeros((cur_num,self.latent_dim+64),device=self.device)
        self.class_temp_cov = torch.zeros(((cur_num,self.latent_dim+64,self.latent_dim+64)),device=self.device)


        if self.trainer.current_epoch == 21 and self.finetune:
            print("Adding encoder parameters")
            for param in self.encoder.parameters():
                param.requires_grad = True
            self.trainer.accelerator_backend.setup_optimizers(self)



    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        z,y,x_hat,x,_,_,_,_ = self.test_step(batch, batch_idx)
        return z,y,x_hat,x

    def classify_data(self,data):
        parr = torch.zeros(data.shape[0],self.class_num)
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(data)
            probs[probs < self.class_bounds[i]] = -10**10
            parr[:,i] = probs
        values,labels = torch.max(parr,dim=1)
        labels[values < -10**9] = self.class_num
        print(len(labels))
        print((labels == self.class_num).sum())
        return labels.to(self.device)


    def get_data_top_val(self,data,all_sift_distances,all_labels):
        parr = torch.zeros(data.shape[0],self.class_num)
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(data)
            parr[:,i] = probs
        max_num = parr.max()+1
        for i in range(self.class_num):
            probs = self.class_gaussian[i].log_prob(data)
            probs -= max_num
            parr[:,i] = probs*(all_sift_distances)
        # parr[:,self.class_num] /= 1000
        row_sum = torch.logsumexp(parr,1)
        values,labels = torch.max(parr,dim=1)
        pvals = torch.exp(values-row_sum)
        # values *=1-pvals
        return values,labels

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

    def log_test_statistics(self,values,pred_labels,labels):
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
        self.log(f"Acc Pos (no unkown det):", float(np.sum(pos_pred_labels == pos_labels)/len(pos_labels)))
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
        self.log(f"Best f1 score (threshold:{thresholds[best_t]}) mistakes_as_open:{misclassification_as_openset[best_t]}:", np.max(f1_scores))
        fig = plt.figure()
        plt.plot(thresholds, f1_scores, color='darkorange')
        plt.xlim([np.min(thresholds), np.max(thresholds)])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Threshold')
        plt.ylabel('F1')
        plt.title('F1 per threshold')
        # self.trainer.logger.experiment.log_image('F1 Curve', fig, description=f'F1 Curve')
        self.trainer.logger.experiment.add_figure('F1 Curve', fig)
        df = pd.DataFrame()
        df['Threshold'] = thresholds
        df['F1'] = f1_scores
        df['Pos acc'] = pos_acc
        df['Neg acc'] = neg_acc
        df.to_csv(f'output/{self.enc_type}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}/Threshold.csv')




    def plot_roc_curve(self,all_data,labels,all_sift_distances):
        all_data = torch.tensor(all_data,device=self.device)
        values,pred_labels = self.get_data_top_val(all_data,all_sift_distances,labels)#这里会减掉最大的logit所以ood的数据负的更多
        values = values.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()
        self.log_test_statistics(values,pred_labels,labels)
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
        self.log('CAUROC',aurocNew)
        self.log("fpr-at-tpr-095",fpr_at_095)
        self.log("tpr-at-tpr-095",tpr_at_095)
        self.log("tnr-at-tpr-095",1-fpr_at_095)
        self.log("acc-at-tpr-095",acc_at_095)

        self.log("fpr-at-tnr-090",fpr_at_tn_090)
        self.log("tnr-at-tnr-090",1-fpr_at_tn_090)
        self.log("tpr-at-tnr-090",tpr_at_tn_090)
        self.log("acc-at-tnr-090",acc_at_tn_090)
        print(f"known_dataset:",self.known_dataset)
        print(f"unknown_dataset:",{self.unknown_dataset})
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
        plt.xlim([-0.00099, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc Curve')
        # self.trainer.logger.experiment.log_image('ROC', fig, description=f'ROC TEST')
        self.trainer.logger.experiment.add_figure('ROC', fig)
        self.log('ROC-AUC-Score',auc(fpr,tpr))
        df = pd.DataFrame()
        df['Threshold'] = thresholds
        df['FPR'] = np.array(fpr)
        df['TPR'] = np.array(tpr)
        df['acc'] = np.array(acc_list)
        # log_table('FPR_TPR', df,experiment=self.trainer.logger.experiment) #这个也没法改
        df.to_csv(f'output/{self.enc_type}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}/FPR_TPR.csv')

    def test_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        pathlib.Path(f'output/{self.enc_type}/{self.known_dataset}_{self.unknown_dataset}/version_{self.trainer.logger._version}').mkdir(parents=True,exist_ok=True)
        all_data = torch.vstack([x[6] for x in outputs]).cpu().numpy()
        all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()    #true label
        if self.enc_type == 'resnet18_semantic_temp_change_position':
            all_logits = torch.vstack([x[7] for x in outputs])
            probs = F.softmax(all_logits,dim=1)
            neg_log_probs = -torch.log2(probs)
            entropy = (probs*neg_log_probs).sum(dim=1)
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
            self.calculate_gaussian()
            self.is_tested = 1
            # np.save("all_data_train.npy",all_data)
            # np.save("all_labels_train.npy",all_labels)
            if self.log_tsne:
                # self.trainer.logger.experiment.log_image('Train T-SNE', fig, description=f'Final TEST')
                self.trainer.logger.experiment.add_figure('Train T-SNE, Final TEST', fig)
            return
        elif self.is_tested == 1:
            self.class_gaussian = self.early_gaussians

        if self.log_tsne:
            # self.trainer.logger.experiment.log_image('T-SNE Without Unknown', fig, description=f'Final TEST')
            self.trainer.logger.experiment.add_figure('T-SNE Without Unknown', fig)
        if self.enc_type == 'resnet18_semantic_temp_change_position':
            pass
        ###
        # all_sift = torch.vstack([x[4].reshape(x[4].shape[0], -1) for x in outputs])
        # all_sift_hat = torch.vstack([x[5].reshape(x[5].shape[0], -1) for x in outputs])
        # all_sift_distances = ((all_sift - all_sift_hat) ** 2).mean(axis=1)
        # all_sift_distances /= all_sift_distances.max()
        # all_sift_distances = all_sift_distances
        ###
        all_sift = torch.vstack([x[3].reshape(x[3].shape[0], -1) for x in outputs])
        all_sift_hat = torch.vstack([x[2].reshape(x[2].shape[0], -1) for x in outputs])
        all_sift_distances = ((all_sift - all_sift_hat) ** 2).mean(axis=1)
        all_sift_distances /= all_sift_distances.max()
        all_sift_distances = all_sift_distances


        # if self.ae_features == False:  # 这个地方对性能影响巨大，看看能不能换成其他的
        #     all_sift_distances[:] = 1
        # all_sift_distances = torch.ones_like(entropy)

        self.class_gaussian = self.early_gaussians
        self.plot_roc_curve(all_data,all_labels,all_sift_distances) #all_data是l_feature

        neg_samples_count = float(np.sum(all_labels == self.class_num))
        pos_samples_count = float(np.sum(all_labels < self.class_num))
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
            self.trainer.logger.experiment.add_figure('T-SNE With Unknown', fig)
            print("end tsne")
        print(self.trainer.logger._version)
        print(self.trainer.logger._version)
        print(self.trainer.logger._version)



    def validation_epoch_end(self, validation_step_outputs):
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
        self.class_counts = torch.zeros(self.class_num,device=self.device)
        self.class_temp_means = torch.zeros((self.class_num,self.latent_dim+64),device=self.device)
        self.class_temp_cov = torch.zeros(((self.class_num,self.latent_dim+64,self.latent_dim+64)),device=self.device)
        self.class_means = torch.zeros((self.class_num,self.latent_dim+64),device=self.device)
        self.class_cov = torch.zeros(((self.class_num,self.latent_dim+64,self.latent_dim+64)),device=self.device)
        self.class_bounds = torch.zeros(self.class_num,device=self.device)

        ## Earlier resnet layers
        self.layer_size = 0
        if self.enc_type == 'densenet':
            self.layer_size += 600
        elif self.enc_type == 'vgg':
            self.layer_size += 640
        elif self.enc_type == 'resnet18_semantic':
            self.layer_size += 448
        elif self.enc_type == 'resnet18_semantic_1':
            self.layer_size += 448
        elif self.enc_type == 'wresnet':
            self.layer_size += 1120
        elif self.enc_type == 'resnet18_semantic_temp_change_position':
            # self.layer_size += 1540+32
            # self.layer_size += 1540+32+60# +60因为把4，4，4变成4的那里让我改回去了
            self.layer_size += 1760
        else:
            self.layer_size += 300
        self.early_temp_means = torch.zeros((self.class_num,self.layer_size),device=self.device)
        self.early_temp_cov = torch.zeros(((self.class_num,self.layer_size,self.layer_size)),device=self.device)
        self.early_means = torch.zeros((self.class_num,self.layer_size),device=self.device)
        self.early_cov = torch.zeros(((self.class_num,self.layer_size,self.layer_size)),device=self.device)

    def configure_optimizers(self):
        #Little hack to get it after device was configured (which is not in init)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #params_list = list(self.encoder.params())+list(self.fc_mu.params())+list(self.fc_var.params)
        finetune_params = list(list(self.fc_mu.parameters())+list(self.fc_var.parameters()))
        params = finetune_params if self.finetune and self.trainer.current_epoch < 20 else self.parameters()
        # print(len(params))
        if self.finetune and self.trainer.current_epoch < 20:
            lr = self.finetune_lr
        # elif self.trainer.current_epoch < self.margin_max_distance and not self.finetune:
        #     lr = self.finetune_lr / 10
        else:
            lr=self.lr
        if self.opt == 'sgd':
            print('SGD')
            optimizer = torch.optim.SGD(params, lr=lr,momentum=0.9, dampening=0, weight_decay=1e-4,nesterov=True)
        else:
            print('ADAM')
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
            self.class_temp_means = torch.zeros((all_classes,self.latent_dim+64),device=self.device)
            self.class_temp_cov = torch.zeros(((all_classes,self.latent_dim+64,self.latent_dim+64)),device=self.device)
            self.class_means = torch.zeros((all_classes,self.latent_dim+64),device=self.device)
            self.class_cov = torch.zeros(((all_classes,self.latent_dim+64,self.latent_dim+64)),device=self.device)
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
        # parser.add_argument("--noise",type=float,default=None)
        # parser.add_argument("--transmit_dim",type=int,required=True)
        return parser



    def calculate_gaussian(self):
            '''
            This is used in testing, to run a training epoch (with no gradients) so we can calculate the gaussians.
            '''
            class_gaussians = []
            early_gaussians = []
            for i in range(self.class_num):
                self.class_means[i,:] = self.class_temp_means[i,:]/self.class_counts[i]
                self.class_cov[i,:,:] = self.class_temp_cov[i,:,:]/self.class_counts[i] - torch.matmul(self.class_means[i,:].view(1,-1).T,self.class_means[i,:].view(1,-1))
                # self.class_cov[i,:,:] += torch.eye(self.class_cov[i,:,:].shape[0],device=self.device)*0.000002
                self.class_cov[i,:,:] = torch.eye(self.class_cov[i,:,:].shape[0],device=self.device)*10
                self.early_means[i,:] = self.early_temp_means[i,:]/self.class_counts[i]
                self.early_cov[i,:,:] = self.early_temp_cov[i,:,:]/self.class_counts[i] - torch.matmul(self.early_means[i,:].view(1,-1).T,self.early_means[i,:].view(1,-1))
                self.early_cov[i,:,:] = (self.early_cov[i,:,:] + torch.t(self.early_cov[i,:,:]))/2
                # self.early_cov[i,:,:] += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001
                # self.early_cov[i,:,:] += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.0001
                try:
                    dist = torch.distributions.multivariate_normal.MultivariateNormal(self.class_means[i,:],self.class_cov[i,:,:])
                except:
                    dist = self.class_gaussian[i]
                class_gaussians.append(dist)
            increases = 4900
            for i in range(self.class_num):
                class_increase = increases
                cov_copy = torch.clone(self.early_cov[i,:,:])
                cov_copy += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001*increases
                while True:
                    try:
                        # torch.cholesky(cov_copy)
                        torch.distributions.multivariate_normal.MultivariateNormal(self.early_means[i,:],cov_copy)
                        break
                    except:
                        cov_copy += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001
                        class_increase += 1
                if class_increase > increases:
                    increases = class_increase
            increases+=2
            print("Increases:", increases)
            for i in range(self.class_num):
                self.early_cov[i,:,:] += torch.eye(self.early_cov[i,:,:].shape[0],device=self.device)*0.000001*increases
                early_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.early_means[i,:],self.early_cov[i,:,:])
                early_gaussians.append(early_dist)

            early_gaussian_mean = self.early_means.sum(axis=0) / self.class_counts.sum()
            early_gaussian_cov = self.early_cov.sum(axis=0) / self.class_counts.sum() - torch.matmul(early_gaussian_mean.view(1,-1).T,early_gaussian_mean.view(1,-1))
            early_gaussian_cov = torch.eye(early_gaussian_cov.shape[0],device=self.device)*1
            early_gaussian = torch.distributions.multivariate_normal.MultivariateNormal(early_gaussian_mean, early_gaussian_cov)
            # early_gaussians.append(early_gaussian)
            self.unfreeze()
            self.class_gaussian = class_gaussians
            self.early_gaussians = early_gaussians
