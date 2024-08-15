import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from models.networks import SIMINet
from models.target_models import target_distribution
from tqdm import tqdm
from utils.annealing import annealing
from utils.parse_config import parse_config
import logging
from utils.kernels import *


class KernelSIVI_langevin(object):
    def __init__(self, config):
        self.config = config
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
        self.target_model = target_distribution[self.target](num_interval = self.config.num_interval, num_obs = self.config.num_obs, beta = self.config.beta, T = self.config.T, sigma = self.config.sigma, device = self.device)
    def learn(self):
        self.preprocess()
        self.loaddata()

        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)
        
        optimizer_VI = torch.optim.Adam([{'params':self.SemiVInet.mu.parameters(),'lr': self.trainpara.lr_SIMI},
                              {'params':self.SemiVInet.log_var,'lr': self.trainpara.lr_SIMI_var}], betas=(.9, .99))
        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)
    
        loss_list = []
        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
            if (epoch-1) ==0:
                X = self.SemiVInet.sampling(num=self.config.sampling.num)
                figname = f'{self.iter_idx+1}.jpg'
                self.target_model.trace_plot(X, figpath=self.save_samples_to_path, figname=figname)
            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                Z_aux = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                X, neg_score_implicit = self.SemiVInet(Z)
                X_aux, neg_score_implicit_aux = self.SemiVInet(Z_aux)
                compu_targetscore = self.target_model.score(X) * annealing_coef(self.iter_idx)
                compu_targetscore_aux = self.target_model.score(X_aux) * annealing_coef(self.iter_idx)

                x_xaux_dist = (compu_targetscore + neg_score_implicit)[:,None,:] * ((compu_targetscore_aux  + neg_score_implicit_aux)[None,:,:])
                loss_kxy = ((x_xaux_dist.sum(-1)) * self.kernel(X,X_aux)).mean()

                if epoch < 0:
                    loss_logp = (self.target_model.logp(X)).mean()
                    loss_kxy = loss_kxy - loss_logp
                optimizer_VI.zero_grad()
                loss_kxy.backward()
                optimizer_VI.step()
                scheduler_VI.step()

                
            # compute some object in the trainging
            loss_list.append(np.array([self.iter_idx, loss_kxy.item()]))    
            logger.info(("Epoch [{}/{}], iters [{}], loss: {:.4f}, net_log_var: {:.4f}").format(epoch, self.trainpara.num_epochs, self.iter_idx, loss_kxy, self.SemiVInet.log_var.mean().item()))
            if epoch%self.config.sampling.visual_time ==0:
                X = self.SemiVInet.sampling(num = self.config.sampling.num)
                figname = f'{self.iter_idx+1}.jpg'
                self.target_model.trace_plot(X, figpath = self.save_samples_to_path, figname = figname)
        loss_list = np.array(loss_list)
        X = self.SemiVInet.sampling(num = self.config.sampling.num)
        torch.save(X.cpu().numpy(), os.path.join(self.save_samples_to_path,'sample{}.pt'.format(self.config.sampling.num)))
        torch.save(loss_list, os.path.join(self.config.log_path,'loss_list.pt'))
        torch.save(self.SemiVInet.state_dict(), os.path.join(self.config.log_path,"SemiVInet.ckpt"))
        return loss_list

if __name__ == "__main__":
    seednow = 2023
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    random.seed(seednow)
    torch.backends.cudnn.deterministic = True

    config = parse_config()

    datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if config.log_stick else "Now_vanilla"
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
    task = KernelSIVI_langevin(config)
    task.learn()