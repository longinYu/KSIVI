import argparse
import os
from datetime import datetime

import numpy as np
import torch

from models.networks import SIMINet
from models.target_models import target_distribution
from tqdm import tqdm
from utils.annealing import annealing
from utils.parse_config import parse_config
import logging

from utils.ema import EMA
from sklearn.model_selection import train_test_split
from utils.kernels import *



class KernelSIVI_bnn(object):
    def __init__(self, config):
        self.config = config

        self.device = self.config.device
        self.target = self.config.target_score
        self.trainpara = self.config.train
        self.num_iters = self.trainpara.num_perepoch * self.config.train.num_epochs
        self.iter_idx = 0
        self.kernel = {"gaussian":gaussian_kernel, "laplace":laplace_kernel}[self.config.kernel]
        
        
    def preprocess(self):
        self.save_samples_to_path = os.path.join(self.config.log_path, "traceplot")
        os.makedirs(self.save_samples_to_path,exist_ok=True)
        
    def loaddata(self):
        # load the datasets
        # The hyperparameters loggamma and loglambda are selected  by MCMC method, like SGLD [1] or SVGD [2].
        # [1] Welling, Max, and Yee W. Teh. "Bayesian learning via stochastic gradient Langevin dynamics." Proceedings of the 28th international conference on machine learning (ICML-11). 2011.
        # [2] Liu, Qiang, and Dilin Wang. "Stein variational gradient descent: A general purpose bayesian inference algorithm." Advances in neural information processing systems 29 (2016). 

        if self.config.target_score == "Bnn_boston":
            data = np.loadtxt('datasets/boston_housing.txt')
            self.loglambda_hyp = -1.003869799168037
            self.loggamma_hyp = -2.555990767319021
        elif self.config.target_score == "Bnn_concrete":
            data = np.loadtxt('datasets/Concrete_Data.csv', delimiter=",")
            self.loglambda_hyp = -0.814071878103707
            self.loggamma_hyp = -3.449819990147337
        elif self.config.target_score == "Bnn_power":
            data = np.loadtxt('datasets/power.csv', delimiter=",")
            self.loglambda_hyp = -0.901769000892788
            self.loggamma_hyp = -2.880246003841932
        elif self.config.target_score == "Bnn_winered":
            data = np.loadtxt('datasets/winered.csv', delimiter=";")
            self.loglambda_hyp = -0.857776388378304
            self.loggamma_hyp = 1.056112577113661
        elif self.config.target_score == "Bnn_protein":
            data = np.loadtxt('datasets/protein.csv', delimiter=",")
            self.loglambda_hyp = -0.788991274446742
            self.loggamma_hyp = -3.161733709183021
        elif self.config.target_score == "Bnn_yacht":
            self.loglambda_hyp = -0.768671081013136
            self.loggamma_hyp = 0.329742800537996
            data = np.loadtxt('datasets/yacht.csv', delimiter=" ")
            
        X_input = torch.from_numpy(data[ :, range(data.shape[1] - 1) ]).to(self.device).float()
        y_input = torch.from_numpy(data[ :, data.shape[1] - 1 ]).to(self.device).float()
        
        ## select the permutaion by train_test_split
        train_ratio = 0.9
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size= 1-train_ratio, random_state=42)

        # from utils.svgd_bnn_hyperparam import svgd_bayesnn
        # logger.info("run svgd ...")
        # svgd = svgd_bayesnn(X_train.cpu().numpy(), y_train.cpu().numpy(), M = 100, batch_size = 100, n_hidden = 50, max_iter = 2000, master_stepsize = 1e-3)
        # self.loggamma_hyp = np.mean(svgd.theta[:,-2])
        # self.loglambda_hyp = np.mean(svgd.theta[:,-1])
        # logger.info("loglambda_hyp: {:.15f}, loggamma_hyp: {:.15f}".format(self.loglambda_hyp, self.loggamma_hyp))
        # rmse_svgd, llk_svgd = svgd.evaluation(X_test.cpu().numpy(), y_test.cpu().numpy())
        # logger.info("svgd: rmse {:.4f}, llk {:.4f}".format(rmse_svgd, llk_svgd))

        y_train = y_train[:,None]
        y_test = y_test[:,None]

        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

        X_train_mean = X_train.mean(0)
        y_train_mean = y_train.mean(0)
        X_train_std = X_train.std(0)
        y_train_std = y_train.std(0)
        
        self.X_train = (X_train - X_train_mean)/X_train_std
        self.y_train = (y_train - y_train_mean)/y_train_std
        self.X_test = (X_test - X_train_mean)/X_train_std
        self.y_test = y_test
        self.X_dev = (X_dev - X_train_mean)/X_train_std
        self.y_dev = y_dev

        self.y_train_mean = y_train_mean
        self.y_train_std = y_train_std
        
        self.size_train = X_train.shape[0]
        self.scale_sto = self.X_train.shape[0]/self.trainpara.sto_batchsize

    def learn(self, model_selection = False):
        self.preprocess()
        self.loaddata()

        self.target_model = target_distribution[self.target](self.device, self.X_train.shape[1], loglambda = self.loglambda_hyp, loggamma = self.loggamma_hyp)
        self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
        
        annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)
        optimizer_VI = torch.optim.Adam([{'params':self.SemiVInet.mu.parameters(),'lr': self.trainpara.lr_SIMI},
                              {'params':self.SemiVInet.log_var,'lr': self.trainpara.lr_SIMI_var}])
        scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)
        
        '''initialize the phinet'''
        optimizer_pre = torch.optim.Adam([{'params':self.SemiVInet.mu.parameters(),'lr': self.trainpara.pre_lr},
                            {'params':self.SemiVInet.log_var,'lr': self.trainpara.pre_lr}])
        for epoch in range(1, self.trainpara.pre_iters_num):
            Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
            X, _ = self.SemiVInet(Z)

            # Use the X_dev dataset to initialize phinet
            predicty_z = self.target_model.predict_y(X, self.X_dev, self.y_train_mean, self.y_train_std)
            loss = ((predicty_z.mean(0) - self.y_dev)**2).mean()
            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()
        ema = EMA(beta=self.trainpara.ema, model_params=self.SemiVInet.parameters())
        for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
            
            for i in range(1, self.trainpara.num_perepoch+1):
                self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i
                batch_idexseq = [bat % self.size_train for bat in range((self.iter_idx - 1) * self.trainpara.sto_batchsize, self.iter_idx * self.trainpara.sto_batchsize)]
                batch_X = self.X_train[batch_idexseq,:]
                batch_y = self.y_train[batch_idexseq,:]
                if not self.trainpara.ustat:
                    Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    Z_aux = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    X, neg_score_implicit = self.SemiVInet(Z)
                    X_aux, neg_score_implicit_aux = self.SemiVInet(Z_aux)
                    compu_targetscore = self.target_model.score(X, batch_X, batch_y, self.scale_sto) * annealing_coef(self.iter_idx)
                    compu_targetscore_aux = self.target_model.score(X_aux, batch_X, batch_y, self.scale_sto) * annealing_coef(self.iter_idx)
                    x_xaux_dist = torch.matmul((compu_targetscore + neg_score_implicit),(compu_targetscore_aux  + neg_score_implicit_aux).T)
                    
                    if not self.trainpara.reg_logp:
                        loss_kxy = (x_xaux_dist * self.kernel(X,X_aux)).mean()
                    else:
                        loss_kxy_ori = (x_xaux_dist * (self.kernel(X,X_aux))).mean() 
                        loss_logp = (self.target_model.logp(X, batch_X, batch_y, self.scale_sto)).mean()
                        loss_kxy = loss_kxy_ori - loss_logp # yacht, winereds
                        # loss_kxy = loss_kxy_ori - loss_logp/100 #  / X.shape[0]
                else:
                    Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    X, neg_score_implicit = self.SemiVInet(Z)
                    compu_targetscore = self.target_model.score(X, batch_X, batch_y, self.scale_sto) * annealing_coef(self.iter_idx)
                    x_xaux_dist = torch.matmul((compu_targetscore + neg_score_implicit),(compu_targetscore  + neg_score_implicit).T)
                    x_xaux_dist.fill_diagonal_(0)

                    if not self.trainpara.reg_logp:
                        loss_kxy = (x_xaux_dist * self.kernel(X,X)).mean()
                    else:
                        loss_kxy_ori = (x_xaux_dist * (self.kernel(X,X))).mean() 
                        loss_logp = (self.target_model.logp(X, batch_X, batch_y, self.scale_sto)).mean()
                        loss_kxy = loss_kxy_ori - loss_logp # yacht, winered
                        # loss_kxy = loss_kxy_ori - loss_logp/100 #  / X.shape[0]
                
                optimizer_VI.zero_grad()
                loss_kxy.backward()
                if self.trainpara.gradient_clip:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.SemiVInet.parameters(), max_norm=1)
                else:
                    gradient_norm = torch.zeros(1)
                optimizer_VI.step()
                scheduler_VI.step()

                ema.update_params(model_parameters=self.SemiVInet.parameters())
                
            if epoch%self.config.sampling.visual_time ==0 or epoch == self.trainpara.num_epochs:
                ema.store(model_parameters=self.SemiVInet.parameters())
                ema.apply_shadow(model_parameters=self.SemiVInet.parameters())
                
                X = self.SemiVInet.sampling(num = self.config.sampling.num)
                with torch.no_grad():
                    test_rmse, test_loglik = self.target_model.rmse_llk(X.to(self.device), self.X_test, self.y_test, self.y_train_mean, self.y_train_std)
                logger.info(("Epoch [{}/{}], loss: {:.4f}, gradient_norm: {:.4f}, test_loglik {:.4f}, rmse {:.4f}").format(epoch, self.trainpara.num_epochs, loss_kxy, gradient_norm.item(), test_loglik, test_rmse))
                ema.restore(model_parameters=self.SemiVInet.parameters())
        # ## Model selection
        # if model_selection:
        #     ema.store(model_parameters=self.SemiVInet.parameters())
        #     ema.apply_shadow(model_parameters=self.SemiVInet.parameters())
        #     X = self.SemiVInet.sampling(num = self.config.sampling.num)
        #     self.target_model.model_selection(X, self.X_dev, self.y_dev, self.y_train_mean, self.y_train_std)
        #     test_rmse, test_loglik = self.target_model.rmse_llk(X, self.X_test, self.y_test, self.y_train_mean, self.y_train_std)
        #     logger.info(("######### After selection, test_loglik {:.4f}, rmse {:.4f}").format(test_loglik, test_rmse))
        
        ema.apply_shadow(model_parameters=self.SemiVInet.parameters())
        torch.save(self.SemiVInet.state_dict(), os.path.join(self.config.log_path,"SemiVInet.ckpt"))
        

if __name__ == "__main__":
    seednow = 2023
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    torch.backends.cudnn.deterministic = True

    config = parse_config()

    datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if config.log_stick else "Now"
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
    task = KernelSIVI_bnn(config)
    task.learn()

    
    


