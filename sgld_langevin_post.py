import torch
import numpy as np
from tqdm import tqdm
from models.target_models import Langevin_post
import os


def SGLD_sampler(score_fn, loop, Z, epsilon_0, trace_plot_fn = False, alpha = 0, check_frq = 500):
    for t in tqdm(range(0, loop)):
        compu_targetscore = score_fn(Z)
        learn_rate = epsilon_0/(1+t)**alpha
        Z = Z + learn_rate/2 * compu_targetscore + np.sqrt(learn_rate) * torch.randn_like(Z).to(device)
        Z = (Z.detach()).clamp(min=-10,max=10) # 很有用.
        if (t+1) % check_frq == 0 and trace_plot_fn:
            figname = f'{t+1}.jpg'
            trace_plot_fn(Z, figname)
    return Z.cpu()


if __name__ == "__main__":
    ## 100, 20 paper
    seed_now = 2022
    torch.manual_seed(seed_now)
    np.random.seed(seed_now)
    num_interval = 100
    num_obs = 20
    beta = 10.0
    T = 1.0
    sigma = 0.1

    ## 50, 10 ablation
    # seed_now = 2022
    # torch.manual_seed(seed_now)
    # np.random.seed(seed_now)
    # num_interval = 50
    # num_obs = 10
    # beta = 10.0
    # T = 1.0
    # sigma = 0.1

    ## 50, 10 ablation
    # seed_now = 2022
    # torch.manual_seed(seed_now)
    # np.random.seed(seed_now)
    # num_interval = 200
    # num_obs = 40
    # beta = 10.0
    # T = 1.0
    # sigma = 0.1

    save_path = os.path.join("SGLD_trace", "langevin_post_{}_obs_{}_interval_{}".format(seed_now,num_obs, num_interval))
    os.makedirs(save_path,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    langevin_model = Langevin_post(num_interval = num_interval, num_obs = num_obs, beta = beta, T = T, sigma=sigma, device = device)
    Z = torch.randn(1000, num_interval).to(device)
    score_fn = lambda x: langevin_model.score(x)
    trace_plot_fn = lambda x, figname: langevin_model.trace_plot(x, figpath = save_path, figname = figname)
    loop = 100000
    epsilon_0 = 1e-4
    trace_SGLD = SGLD_sampler(score_fn, trace_plot_fn = trace_plot_fn, loop = loop, Z = Z, epsilon_0 = epsilon_0, alpha = 0)
    torch.save(trace_SGLD, os.path.join(save_path, "parallel_SGLD_langevin_sgld_{}_obs_{}_interval_{}_loop_{}_step_{}.pt".format(seed_now, num_obs, num_interval, loop, epsilon_0)))