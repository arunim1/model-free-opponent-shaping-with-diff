import optuna
from optuna_dashboard import run_server
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing
import json

import torch
from environments import NonMfosMetaGames
import subprocess
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore

def worker(args):
    game, nn_game, ccdr, p1, p2, lr, num_steps, batch_size, device = args
    curr_nn_game = nn_game and game.find("diff") != -1 
    env = NonMfosMetaGames(batch_size, p1=p1, p2=p2, game=game, lr=lr, asym=None, nn_game=curr_nn_game, ccdr=ccdr)
    env.reset()
    M_mean = torch.zeros(4).to(device)

    for i in range(num_steps):
        params, r0, r1, M = env.step()
        M_mean += M.detach().mean(dim=0).squeeze() / M.detach().mean(dim=0).squeeze().sum()

    return M_mean / num_steps

def def_objective(in_game, in_nn_games=[True]):
    global the_game
    the_game = in_game
    global the_nn_games
    the_nn_games = in_nn_games
    def objective(trial):
        lr1 = trial.suggest_float("lr1", 0, 10)
        lr2 = trial.suggest_float("lr2", 0, 10)
        lr = (lr1, lr2)
        
        batch_size = 64 # 4096
        num_steps = 100 # 100
    
        # base_games= ["PD", "MP", "HD", "SH"]
        base_games= [the_game]

        diff_oneshot = ["diff" + game for game in base_games]
        iterated = ["I" + game for game in base_games]
        diff_iterated = ["diffI" + game for game in base_games]
        diff_games = diff_oneshot + diff_iterated
        all_games = diff_iterated + iterated + diff_oneshot + base_games

        p1s = ["NL", "LOLA"]
        p2s = ["NL", "LOLA"]

        # Combine the lists and find all unique combinations
        pairs = product(p1s, p2s)
        pairs = list(set(tuple(sorted(pair)) for pair in pairs))

        ccdrs = [False]
        nn_games = the_nn_games

        total_M_mean = 0 # or 0 if additive
        tasks = []
        actual_games = diff_games if nn_games[0] else all_games

        for game in actual_games: 
            if game in diff_games: nn_games_2 = nn_games
            else: nn_games_2 = [False]
            for nn_game in nn_games_2: 
                for ccdr in ccdrs: 
                    for p1, p2 in pairs:
                        tasks.append((game, nn_game, ccdr, p1, p2, lr, num_steps, batch_size, device))
        
        with Pool() as pool:
            results = pool.map(worker, tasks)
            total_M_mean = sum(results) / len(results)
        print(total_M_mean)

        return total_M_mean[0].item()
    
    return objective

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    storage = optuna.storages.InMemoryStorage()
    base_games= ["PD"]#, "HD", "SH"]
    nn_games = [[True]]#, [False]]
    for in_game in base_games:
        for nn_game in nn_games:
            nn = "nn" if nn_game[0] else "no_nn"
            objective = def_objective(in_game, nn_game)
            study = optuna.create_study(study_name=f'{in_game}_{nn}_lr_tuning', storage=storage, direction='maximize')
            study.optimize(objective, n_trials=50)

            # Extract the results you want
            results = [{'trial': trial.number, 'value': trial.value, 'params': trial.params} for trial in study.trials]

            # Save to JSON
            if not os.path.exists('runs/lr_tuning'):
                os.makedirs('runs/lr_tuning')
            json_file_path = f'runs/lr_tuning/{in_game}_{nn}.json'
            with open(json_file_path, 'w') as f:
                json.dump(results, f)

    run_server(storage)

