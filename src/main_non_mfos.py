import os
import json
import torch
import argparse
from environments import NonMfosMetaGames
import subprocess
from tqdm import tqdm
from itertools import product

from tuning_plotting import main as plot_lr_tuning
from asymm_plotting_full import main as plot_asymm
from plot_bigtime import main as plot_all

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="")
args = parser.parse_args()

if __name__ == "__main__":
    torch.cuda.empty_cache()

    batch_size = 1024 # 4096
    num_steps = 1000 # 100
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(f"runs/{name}"):
        os.mkdir(f"runs/{name}")
        with open(os.path.join(f"runs/{name}", "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore
    
    results = []
    base_games= ["SL"]#, "MP", "HD", "SH"]

    diff_oneshot = ["diff" + game for game in base_games]
    iterated = ["I" + game for game in base_games]
    diff_iterated = ["diffI" + game for game in base_games]
    diff_games = diff_oneshot + diff_iterated
    all_games = diff_iterated + iterated + diff_oneshot + base_games
    all_games = diff_oneshot
    p1s = ["NL", "LOLA"]
    p2s = ["NL", "LOLA"]
    pairs = product(p1s, p2s)
    pairs = list(set(tuple(sorted(pair)) for pair in pairs))

    # list lrs from 0.001 to 1.0, with a few in between each power of 10
    lrs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2., 5., 10.]
    asymmetries = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ccdrs = [True, False]
    nn_games = [True, False]

    lrs = [(7.904258320966191, 9.48270678649406)]
    asymmetries = [None]
    ccdrs = [False]
    nn_games = [True]

    for game in all_games: 
        for nn_game in nn_games: 
            for ccdr in ccdrs: 
                for lr in lrs: 
                    for asym in asymmetries: 
                        for p1, p2 in pairs:
                            curr_nn_game = nn_game and game in diff_games 
                            env = NonMfosMetaGames(batch_size, p1=p1, p2=p2, game=game, lr=lr, asym=asym, nn_game=curr_nn_game, ccdr=ccdr)
                            env.reset()
                            running_rew_0 = torch.zeros(batch_size).to(device)
                            running_rew_1 = torch.zeros(batch_size).to(device)
                            all_params_1 = []
                            all_Ms = []
                            rewards_1 = []
                            rewards_2 = []
                            last_params = [0]
                            M_mean = torch.zeros(4).to(device)

                            for i in tqdm(range(num_steps)):
                                params, r0, r1, M = env.step() # _, r0, r1, M
                                running_rew_0 += r0.squeeze(-1)
                                running_rew_1 += r1.squeeze(-1)
                                
                                M_1 = M[0, :].detach().tolist() # just the first run of the batch, for now
                                all_Ms.append(M_1)

                                M_mean += M.detach().mean(dim=0).squeeze()
                                
                                # just p1's params
                                params_1 = params[:batch_size, :] 
                                all_params_1.append(params_1[0,:].detach().tolist()) # just the first run of the batch
                                    
                                rewards_1.append(r0.mean(dim=0).squeeze().tolist())
                                rewards_2.append(r1.mean(dim=0).squeeze().tolist())

                                if i == num_steps - 1: last_params[0] = params

                            
                            split_params = torch.split(last_params[0], batch_size, dim=0) 
                            end_params = [split_params[0][:50].tolist(), split_params[1][:50].tolist()]

                            mean_rew_0 = (running_rew_0.mean() / num_steps).item()
                            mean_rew_1 = (running_rew_1.mean() / num_steps).item()
                            
                            results.append({"game": game, 
                                            "p1": p1, 
                                            "p2": p2, 
                                            "rew_0": mean_rew_0, 
                                            "rew_1": mean_rew_1, 
                                            "all_params_1": all_params_1 if not curr_nn_game else None, 
                                            "end_params": end_params if not curr_nn_game else None, 
                                            "all_Ms": all_Ms, 
                                            "rewards_1": rewards_1, 
                                            "rewards_2": rewards_2, 
                                            "lr": lr, 
                                            "eps": asym, 
                                            "ccdr": ccdr,
                                            "nn_game": nn_game,
                                            "M_mean": M_mean.tolist()
                                            })
                            print("=" * 100)
                            print(f"Done with game: {game}, p1: {p1}, p2: {p2}")
                            print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")


    with open(os.path.join(f"runs/{name}", f"out.json"), "w") as f:
        json.dump(results, f)

    print("Running plot_bigtime.py with filename: ", name)

    if len(lrs) > 1:
        plot_lr_tuning(name)
    elif len(asymmetries) > 1:
        plot_asymm(name)
    else:
        plot_all(name, caller="non_mfos")

    # this doesnt end up running because the above function calls also end up doing "quit()"
    # command = ["gcloud", "compute", "instances", "stop", "instance-arunim", "--zone=us-east1-b"]

    # print("Executing command:", " ".join(command))
    # subprocess.run(command, capture_output=False, text=True)
