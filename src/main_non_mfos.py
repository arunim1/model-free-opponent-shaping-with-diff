import os
import json
import torch
import argparse
from environments import NonMfosMetaGames

from plot_bigtime import main as plot_all

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--nn-game", action="store_true", default=False)
args = parser.parse_args()

if __name__ == "__main__":
    batch_size = 4096
    num_steps = 1000 # 500?
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(f"runs/{name}"):
        os.mkdir(f"runs/{name}")
        with open(os.path.join(f"runs/{name}", "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore
    nn_game = args.nn_game

    results = []
    base_games= ["PD", "MP", "HD", "SH"]

    diff_oneshot = ["diff" + game for game in base_games]
    iterated = ["I" + game for game in base_games]
    diff_iterated = ["diffI" + game for game in base_games]
    diff_games = diff_oneshot + diff_iterated
    all_games = base_games + diff_oneshot + iterated + diff_iterated

    p1s = ["NL", "LOLA"]
    p2s = ["NL", "LOLA"]

    for game in all_games:
        for p1 in p1s:
            for p2 in p2s:
                env = NonMfosMetaGames(batch_size, game=game, p1=p1, p2=p2)
                env.reset()
                running_rew_0 = torch.zeros(batch_size).to(device)
                running_rew_1 = torch.zeros(batch_size).to(device)
                all_params_1 = []
                all_Ms = []
                rewards_1 = []
                rewards_2 = []
                last_params = []

                for i in range(num_steps):
                    params, r0, r1, M = env.step() # _, r0, r1, M
                    running_rew_0 += r0.squeeze(-1)
                    running_rew_1 += r1.squeeze(-1)
                    
                    M_1 = M[0, :].detach().tolist() # just the first run of the batch
                    # alternatively, take the mean of the batch
                    # M_1 = M.mean(dim=0).squeeze().tolist()
                    all_Ms.append(M_1)

                    # just p1's params
                    params_1 = params[:batch_size, :] 

                    all_params_1.append(params_1[0,:].detach().tolist()) # just the first run of the batch

                    rewards_1.append(r0.mean(dim=0).squeeze().tolist())
                    rewards_2.append(r1.mean(dim=0).squeeze().tolist())
                    last_params[0] = params
                
                split_params = torch.split(last_params[0], batch_size, dim=0) 
                end_params = [split_params[0][:50].tolist(), split_params[1][:50].tolist()]

                mean_rew_0 = (running_rew_0.mean() / num_steps).item()
                mean_rew_1 = (running_rew_1.mean() / num_steps).item()
                
                results.append({"game": game, 
                                "p1": p1, 
                                "p2": p2, 
                                "rew_0": mean_rew_0, 
                                "rew_1": mean_rew_1,
                                "all_params_1": all_params_1 if not nn_game else None,
                                "end_params": end_params if not nn_game else None,
                                "all_Ms": all_Ms,
                                "rewards_1": rewards_1,
                                "rewards_2": rewards_2,
                                })
                print("=" * 100)
                print(f"Done with game: {game}, p1: {p1}, p2: {p2}")
                print(f"r0: {mean_rew_0}, r1: {mean_rew_1}")

                # =======

    with open(os.path.join(f"runs/{name}", f"out.json"), "w") as f:
        json.dump(results, f)

    print("Running plot_bigtime.py with filename: ", name)

    plot_all(name, caller="non_mfos", nn_game=nn_game)