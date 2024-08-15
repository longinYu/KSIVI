import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.target_models import target_distribution
import scipy.io

def SGLD_lr(loop = 10000, Z = torch.zeros([100, 56]), epsilon_0 = 1e-3, alpha = 0):
    data = scipy.io.loadmat('datasets/waveform.mat')
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    # add scalar column of one.
    X_train = torch.from_numpy(X_train).to(device).float()
    X_test = torch.from_numpy(X_test).to(device).float()
    y_train = torch.from_numpy(y_train).to(device).reshape(-1,1).float()
    y_test = torch.from_numpy(y_test).to(device).reshape(-1,1).float()
    
    Z = Z.to(device)
    trace = []

    for t in tqdm(range(1, loop+1)):
        for i in range(1, num_perepoch+1):
            iters = (t-1) * num_perepoch + i
            batch_X = X_train
            batch_y = y_train
            compu_targetscore = model.score(Z, batch_X, batch_y, 1)
            learn_rate = np.max((epsilon_0 /(iters)**alpha, 1e-8))
            Z = Z + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn([Z.shape[0],Z.shape[1]]).to(device)
        trace.append(Z[0,:].cpu().numpy())
    torch.save(np.array(trace), "SGLD_trace/singletrace_LRwaveform.pt")
    plt.plot(np.array(trace)[:,-1],"-",label = "Last dimension trace")
    plt.legend()
    plt.savefig('SGLD_trace/parallel_SGLD_LRwaveform_test.jpg')
    return Z.cpu()


def MALA_lr(loop = 10000, Z = torch.zeros([100, 56]), epsilon_0 = 1e-3, alpha = 0):
    data = scipy.io.loadmat('datasets/waveform.mat')
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    # add  scalar column of one.
    X_train = torch.from_numpy(X_train).to(device).float()
    X_test = torch.from_numpy(X_test).to(device).float()
    y_train = torch.from_numpy(y_train).to(device).reshape(-1,1).float()
    y_test = torch.from_numpy(y_test).to(device).reshape(-1,1).float()
    
    Z = Z.to(device)
    trace = []
    for t in tqdm(range(1, loop+1)):
        for i in range(1, num_perepoch+1):
            iters = (t-1) * num_perepoch + i
            batch_X = X_train
            batch_y = y_train
            compu_targetscore = model.score(Z, batch_X, batch_y, 1)
            
            learn_rate = np.max((epsilon_0 /(iters)**alpha, 1e-8))
            proposal = Z + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn([Z.shape[0],Z.shape[1]]).to(device)
            
            log_accept_ratio = model.log_likelihood(proposal,batch_X, batch_y, 1) - model.log_likelihood(Z,batch_X, batch_y, 1) + \
                    0.5 / learn_rate * (-torch.norm(Z - proposal - learn_rate/2 * model.score(proposal, batch_X, batch_y, 1),dim=-1)**2 +\
                                        torch.norm(proposal - Z - learn_rate/2 * compu_targetscore,dim=-1)**2)
            accept_ratio = (torch.exp(log_accept_ratio)).clamp(max=1)
            mask = torch.rand(accept_ratio.shape[0]).to(device) <= accept_ratio
            Z[mask] = proposal[mask]
        trace.append(Z[0,:].cpu().numpy())
    plt.plot(np.array(trace)[:,-1],"-",label = "Last dimension trace")
    plt.legend()
    plt.savefig('SGLD_trace/parallel_MALA_LRwaveform.jpg')
    return Z.cpu()



if __name__ == "__main__":
    flag = 1
    seednow = 2022
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sto_batchsize = 400
    num_perepoch = 100
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    torch.backends.cudnn.deterministic = True
    M = 1000
    D = 22
    model = target_distribution["LRwaveform"](device)
    ## SGLD 
    # sample_SGLD = SGLD_lr(loop = 4000, Z = torch.zeros(M, D), epsilon_0 = 1e-4, alpha = 0)
    # torch.save(sample_SGLD, "SGLD_trace/parallel_SGLD_LRwaveform.pt")
    ## MALA
    sample_MALA = MALA_lr(loop = 4000, Z = torch.zeros(M, D), epsilon_0 = 1e-4, alpha = 0)
    torch.save(sample_MALA, "SGLD_trace/parallel_MALA_LRwaveform.pt")