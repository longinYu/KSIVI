import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import scipy.io

from models.networks import SIMINet
from models.target_models import target_distribution
from tqdm import tqdm
from utils.annealing import annealing
from utils.density_estimation import density_estimation
from utils.parse_config import parse_config

import matplotlib.pyplot as plt
import seaborn as sns

import logging
import time
from utils.kernels import *


class KernelSIVI_lr(object):
    def __init__(self, config):
        self.config = config
        self.config.device = "cpu"
        self.device = self.config.device
        self.target = self.config.target_score
        self.trainpara = self.config.train
        self.num_iters = self.trainpara.num_perepoch * self.config.train.num_epochs
        self.iter_idx = 0
        self.kernel = {"gaussian":gaussian_kernel, "laplace":laplace_kernel,"IMQ":IMQ_kernel}[self.config.kernel]
        
    def preprocess(self):
        self.save_samples_to_path = os.path.join(self.config.log_path, "traceplot")
        os.makedirs(self.save_samples_to_path,exist_ok=True)
         
    def loaddata(self):
        # load the datasets
        if self.target in ["LRwaveform"]:
            data = scipy.io.loadmat('datasets/waveform.mat')
            
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            self.X_train = torch.from_numpy(X_train).to(self.device).float()
            self.X_test = torch.from_numpy(X_test).to(self.device).float()
            self.y_train = torch.from_numpy(y_train).to(self.device).reshape(-1,1).float()
            self.y_test = torch.from_numpy(y_test).to(self.device).reshape(-1,1).float()

            self.size_train = X_train.shape[0]
            self.scale_sto = X_train.shape[0]/self.trainpara.sto_batchsize
            self.baseline_sample = torch.load("{}".format(self.config.baseline_sample))

    def learn(self):
        self.preprocess()
        self.loaddata()

        self.target_model = target_distribution[self.target](self.device)
        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)

        optimizer_VI = torch.optim.Adam([{'params':self.SemiVInet.mu.parameters(),'lr': self.trainpara.lr_SIMI},
                              {'params':self.SemiVInet.log_var,'lr': self.trainpara.lr_SIMI_var}])
        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)
    
        loss_list = []
        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                # ============================================================== #
                #                      Train the SemiVInet                       #
                # ============================================================== #
                Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                X, neg_score_implicit = self.SemiVInet(Z)
                Z_aux = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                X_aux, neg_score_implicit_aux = self.SemiVInet(Z_aux)
                

                compu_targetscore = self.target_model.score(X, self.X_train, self.y_train, self.scale_sto) * annealing_coef(self.iter_idx)
                compu_targetscore_aux = self.target_model.score(X_aux, self.X_train, self.y_train, self.scale_sto) * annealing_coef(self.iter_idx)
                x_xaux_dist = torch.matmul((compu_targetscore + neg_score_implicit),(compu_targetscore_aux  + neg_score_implicit_aux).T)
                kernel_dist, kernel_width = self.kernel(X,X_aux, get_width = True)
                loss_kxy = (x_xaux_dist * kernel_dist).mean()
                optimizer_VI.zero_grad()
                loss_kxy.backward()
                
                optimizer_VI.step()
                scheduler_VI.step()
                loss_list.append(np.array([self.iter_idx, loss_kxy.item()]))    
            if epoch% self.config.sampling.visual_time ==0:
                X = self.SemiVInet.sampling(num = self.config.sampling.num)
                if epoch%(self.config.sampling.visual_time) ==0:
                    plt.cla()
                    figpos, axpos = plt.subplots(5, 5,figsize = (15,15), constrained_layout=False)
                    for plotx in range(1,6):
                        for ploty in range(1,6):
                            if ploty != plotx:
                                X1, Y1, Z = density_estimation(X[:,plotx].cpu().numpy(), X[:,ploty].cpu().numpy())
                                axpos[plotx-1,ploty-1].contour(X1, Y1, Z,colors= "#ff7f0e")
                                X1, Y1, Z = density_estimation(self.baseline_sample[:,plotx].cpu().numpy(), self.baseline_sample[:,ploty].cpu().numpy())
                                axpos[plotx-1,ploty-1].contour(X1, Y1, Z,colors= 'black')
                            else:
                                sns.kdeplot(X[:,plotx].cpu().numpy(),fill=True,color= "#ff7f0e",ax = axpos[plotx-1, ploty-1], label="SIVISM").set(ylabel=None)
                                sns.kdeplot(self.baseline_sample[:,plotx].cpu().numpy(),fill=True,color= "black",ax = axpos[plotx-1, ploty-1], label="SGLD").set(ylabel=None)
                                axpos[plotx-1,ploty-1].legend()
                    figpos.tight_layout()
                    save_to_path = os.path.join(self.save_samples_to_path, "sample_scatterplot"+str(self.iter_idx)+'.png')
                    plt.savefig(save_to_path)
                    plt.close()
                    torch.save(X.cpu(), save_to_path.replace(".png",".pt"))
                
                logger.info(("Epoch [{}/{}], iters [{}], loss: {:.4f}").format(epoch, self.trainpara.num_epochs, self.iter_idx, loss_kxy))
                logger.info("compu_targetscore: {:.4f}, neg_score_implicit: {:.4f}, f_norm, {:.4f}, kernel_width: {:.4f}".format(compu_targetscore.mean().item(), neg_score_implicit.abs().mean().item(),(compu_targetscore + neg_score_implicit).mean().item(),kernel_width))
        loss_list = np.array(loss_list)
        X = self.SemiVInet.sampling(num = self.config.sampling.num)
        torch.save(X.cpu().numpy(), os.path.join(self.save_samples_to_path,'sample{}.pt'.format(self.config.sampling.num)))
        torch.save(loss_list, os.path.join(self.config.log_path,'loss_list.pt'))
        torch.save(self.SemiVInet.state_dict(), os.path.join(self.config.log_path,"SemiVInet.ckpt"))
        return loss_list

if __name__ == "__main__":
    seednow = 2022
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    random.seed(seednow)
    torch.backends.cudnn.deterministic = True

    config = parse_config()

    datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if config.log_stick else "vanilla"
    config.log_path = os.path.join("expkernelSIVI", config.target_score, "{}".format(datetimelabel))
    os.makedirs(config.log_path,exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(os.path.join(config.log_path,"final.log"))
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.info('Training with the following settings:')
    for name, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            for name, value in vars(value).items():
                logger.info('{} : {}'.format(name, value))
        else:
            logger.info('{} : {}'.format(name, value))
    config.logger = logger
    task = KernelSIVI_lr(config)
    task.learn()

    ## ablation study of the learning rate
    # for lr in [0.008]:
    #     seednow = 2022
    #     torch.manual_seed(seednow)
    #     torch.cuda.manual_seed_all(seednow)
    #     np.random.seed(seednow)
    #     random.seed(seednow)
    #     torch.backends.cudnn.deterministic = True

    #     config = parse_config()

    #     datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if config.log_stick else f"vanilla_lr_{lr}"
    #     config.log_path = os.path.join("expkernelSIVI", config.target_score, "{}".format(datetimelabel))
    #     os.makedirs(config.log_path,exist_ok=True)

    #     logger = logging.getLogger()
    #     logger.setLevel(logging.INFO)
    #     filehandler = logging.FileHandler(os.path.join(config.log_path,"final.log"))
    #     filehandler.setLevel(logging.INFO)
    #     logger.addHandler(filehandler)
    #     config.train.lr_SIMI = lr
    #     config.train.lr_SIMI_var = lr
    #     logger.info('Training with the following settings:')
    #     for name, value in vars(config).items():
    #         if isinstance(value, argparse.Namespace):
    #             for name, value in vars(value).items():
    #                 logger.info('{} : {}'.format(name, value))
    #         else:
    #             logger.info('{} : {}'.format(name, value))
    #     config.logger = logger
    #     task = KernelSIVI_lr(config)
    #     task.learn()