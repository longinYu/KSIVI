
import torch
import numpy as np

def gaussian_kernel(samples_x, samples_y, h = -1, get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = ((samples_x[:,None,:] - samples_y[None,:,:])**2).sum(-1)
    if h < 0: # use the median trick
        if detach:
            h = torch.median(pairwise_dists).detach()
        else:
            h = torch.median(pairwise_dists)
        h = torch.sqrt(0.5 * h /np.log(samples_x.shape[0] + 1))
    kxy = torch.exp(- pairwise_dists/ h**2 / 2)
    if get_width:
        return kxy, h 
    else:
        return kxy



def laplace_kernel(samples_x, samples_y, h = -1,get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = torch.abs((samples_x[:,None,:] - samples_y[None,:,:])).sum(-1)
    if h < 0: # use the median trick
        h = torch.median(pairwise_dists).detach()
        h = h /np.log(samples_x.shape[0] + 1)
    kxy = torch.exp(- pairwise_dists/ h )
    if get_width:
        return kxy, h 
    else:
        return kxy


def IMQ_kernel(samples_x, samples_y, h = -1, get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = ((samples_x[:,None,:] - samples_y[None,:,:])**2).sum(-1)
    if h < 0: # use the median trick
        h = torch.median(pairwise_dists).detach()
        h = h /np.log(samples_x.shape[0] + 1)
    kxy = (1 + pairwise_dists/ h )**(-0.5)
    if get_width:
        return kxy, h 
    else:
        return kxy


def Riesz_kernel(samples_x, samples_y, h = -1, get_width = False, detach=False):
    '''
    samples_x: shape = [n,d]
    samples_y: shape = [n,d]
    '''
    pairwise_dists = (-(samples_x[:,None,:] - samples_y[None,:,:]).norm(1,dim=-1) 
                      + (samples_x[:,None,:]).norm(1,dim=-1) 
                      + (samples_y[None,:,:]).norm(1,dim=-1))
    if h < 0: # use the median trick
        h = torch.median(pairwise_dists).detach()
        h = h /np.log(samples_x.shape[0] + 1)
    kxy = pairwise_dists/ h
    if get_width:
        return kxy, h 
    else:
        return kxy
