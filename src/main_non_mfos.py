import os
import json
import torch
import argparse
from environments import NonMfosMetaGames
import subprocess
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool
import multiprocessing
from scipy.spatial.distance import pdist, squareform
import numpy as np

from tuning_plotting import main as plot_lr_tuning
from asymm_plotting_full import main as plot_asymm
from plot_bigtime import main as plot_all
from esvvslr import main as plot_esv_vs_lr

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="")
args = parser.parse_args()


def worker(args_in):
    game, nn_game, batch_size, num_steps, p1, p2, lr, asym, ccdr, adam, device = args_in

    curr_nn_game = nn_game and game.find("diff") != -1  
    env = NonMfosMetaGames(batch_size, p1=p1, p2=p2, game=game, lr=lr, asym=asym, nn_game=curr_nn_game, ccdr=ccdr, adam=adam)
    env.reset()
    running_rew_0 = torch.zeros(batch_size).to(device)
    running_rew_1 = torch.zeros(batch_size).to(device)
    all_params_1 = []
    all_Ms = []
    rewards_1 = []
    rewards_2 = []
    last_params = [0]
    last_r0 = 0
    last_r1 = 0
    M_mean = torch.zeros(4).to(device)

    for i in tqdm(range(num_steps)):
        params, r0, r1, M = env.step() # _, r0, r1, M
        running_rew_0 += r0.squeeze(-1)
        running_rew_1 += r1.squeeze(-1)
        
        # just the first 5 runs of the batch
        M_1 = M[:5, :].detach().squeeze().tolist()
        all_Ms.append(M_1)

        M_mean += M.detach().mean(dim=0).squeeze()
        
        # just p1's params
        params_1 = params[:batch_size, :] 
        all_params_1.append(params_1[0,:].detach().tolist()) # just the first run of the batch
            
        rewards_1.append(r0.mean(dim=0).squeeze().tolist())
        rewards_2.append(r1.mean(dim=0).squeeze().tolist())

        if i == num_steps - 1: last_params[0], last_r0, last_r1 = params, r0, r1

    stability_test = False
    if stability_test:
        n_of_perturbations = 10000
        epsilon = lr
        num_improvements_1 = 0
        num_improvements_2 = 0
        for _ in range(n_of_perturbations):
            # altering params randomly: 
            split_params = torch.split(last_params[0], batch_size, dim=0)
            end_params_1 = split_params[0] # just p1's params
            end_params_2 = split_params[1] # just p2's params

            rand_tensor = torch.randn(end_params_1.shape) - 0.5
            norm_rand_tensor = rand_tensor / (rand_tensor.norm(dim=1, keepdim=True)+ 1e-8)
            params_mod_1 = end_params_1 + epsilon*norm_rand_tensor
            rand_tensor = torch.randn(end_params_2.shape) - 0.5
            norm_rand_tensor = rand_tensor / (rand_tensor.norm(dim=1, keepdim=True)+ 1e-8)  
            params_mod_2 = end_params_2 + epsilon*norm_rand_tensor
            _, r0, _, _ = env.fwd_step(params_mod_1, end_params_2)
            _, _, r1, _ = env.fwd_step(end_params_1, params_mod_2)
            if r0.mean().item() > last_r0.mean().item():
                num_improvements_1 += 1
            if r1.mean().item() > last_r1.mean().item():
                num_improvements_2 += 1

        print(f"stability_1 = {num_improvements_1 / n_of_perturbations}")
        print(f"stability_2 = {num_improvements_2 / n_of_perturbations}")
    
    # Collect all runs into a single tensor
    # Reshape your data to a matrix of shape (5, 500*4)
    all_Ms_np = np.array(all_Ms).transpose(1, 0, 2)
    all_Ms_reshaped = all_Ms_np.reshape(5, -1)
    # Compute pairwise Euclidean distances
    distances = pdist(all_Ms_reshaped, 'euclidean')
    # Convert to square form
    square_distances = squareform(distances)
    # Sum the distances for each run
    sum_distances = square_distances.sum(axis=1)
    # Find the index of the most representative run
    most_representative_idx = sum_distances.argmin()
    # Get the most representative run
    most_representative_run = all_Ms_np[most_representative_idx].tolist()
    
    M_mean /= num_steps
    M_mean_list = M_mean.tolist()
    # 2 decimal places
    M_mean_list = [round(x, 2) for x in M_mean_list]
    
    split_params = torch.split(last_params[0], batch_size, dim=0) 
    end_params = [split_params[0][:50].tolist(), split_params[1][:50].tolist()]

    mean_rew_0 = (running_rew_0.mean() / num_steps).item()
    mean_rew_1 = (running_rew_1.mean() / num_steps).item()
    
    result = {"game": game, 
                    "p1": p1, 
                    "p2": p2, 
                    "rew_0": mean_rew_0, 
                    "rew_1": mean_rew_1, 
                    "all_params_1": all_params_1 if not curr_nn_game else None, 
                    "end_params": end_params if not curr_nn_game else None, 
                    "all_Ms": most_representative_run, 
                    "rewards_1": rewards_1, 
                    "rewards_2": rewards_2, 
                    "lr": lr, 
                    "eps": asym, 
                    "ccdr": ccdr,
                    "nn_game": nn_game,
                    "M_mean": M_mean_list
                    }
    print("=" * 100)
    print(f"Done with game: {game}, p1: {p1}, p2: {p2}")
    print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    torch.cuda.empty_cache()

    batch_size = 128 # 4096
    num_steps = 100 # 100
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(f"runs/{name}"):
        os.mkdir(f"runs/{name}")
        with open(os.path.join(f"runs/{name}", "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore
    
    results = []
    base_games= ["PD"]
    diff_oneshot = ["diff" + game for game in base_games]
    iterated = ["I" + game for game in base_games]
    diff_iterated = ["diffI" + game for game in base_games]
    diff_games = diff_oneshot + diff_iterated
    all_games = base_games + iterated + diff_oneshot + diff_iterated
    # all_games = diff_oneshot
    p1s = ["NL", "LOLA"]
    p2s = ["NL", "LOLA"]
    pairs = product(p1s, p2s)
    pairs = list(set(tuple(sorted(pair)) for pair in pairs))

    # list lrs from 0.001 to 1.0, with a few in between each power of 10
    lrs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2., 5., 10.]
    asymmetries = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ccdrs = [False, True]
    nn_games = [True, False]

    # lrs = [(1.62885553603262, 2.220943843335561)]
    lrs =[0.1, 0.2, 0.5, 1, 2, 5, 10]
    lrs = np.logspace(-2, 1, num=20)
    asymmetries = [None]
    # ccdrs = [False]
    nn_games = [True]
    
    # Alternative method: use multiprocessing
    multiprocessing.set_start_method('spawn')

    tasks = []
    for game in diff_games: 
        if game in diff_games: nn_games_2 = nn_games
        else: nn_games_2 = [False] 
        for nn_game in nn_games_2: 
            for ccdr in ccdrs: 
                for asym in asymmetries: 
                    for p1, p2 in pairs:
                        for lr in lrs:
                            adam = True 
                            tasks.append((game, nn_game, batch_size, num_steps, p1, p2, lr, asym, ccdr, adam, device))

    with Pool(6) as pool:
        results = pool.map(worker, tasks)
        if not os.path.isdir(f"runs/{name}"):
            os.mkdir(f"runs/{name}")
        with open(os.path.join(f"runs/{name}", f"out.json"), "w") as f:
            json.dump(results, f)

    print("Running plot_bigtime.py with filename: ", name)

    if len(lrs) > 1:
        for lr in lrs:
            # plot_all(f'{name}_{lr}', caller="non_mfos")
            pass
        plot_esv_vs_lr(name)
    elif len(asymmetries) > 1:
        plot_asymm(name)
    else:
        plot_all(name, caller="non_mfos")


    # command = ["gcloud", "compute", "instances", "stop", "instance-arunim", "--zone=us-east1-b"]

    # print("Executing command:", " ".join(command))
    # subprocess.run(command, capture_output=False, text=True)
