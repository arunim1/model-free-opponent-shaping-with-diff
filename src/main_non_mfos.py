import os
import json
import torch
import argparse
from environments import NonMfosMetaGames
import numpy as np
import time
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="delete")
parser.add_argument("--G", type=float, default=2)
parser.add_argument("--threshold", type=str, default=None)
parser.add_argument("--pwlinear", type=int, default=None)
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_log(
    pms,
    p1,
    p2,
    lr,
    asym,
    threshold,
    pwlinear,
    ccdr,
    adam,
    num_steps,
    batch_size,
    runs_to_track,
    seed,
):
    # initialize empty log for this game
    log = {}
    log["payoff_mat_p2"] = pms[0].cpu().numpy().tolist()
    log["payoff_mat_p1"] = pms[1].cpu().numpy().tolist()
    log["p1"] = p1
    log["p2"] = p2
    log["lr"] = lr
    log["asym"] = asym
    log["threshold"] = threshold
    log["pwlinear"] = pwlinear
    log["ccdr"] = ccdr
    log["adam"] = adam
    log["num_steps"] = num_steps
    log["batch_size"] = batch_size
    log["avg_params_p1"] = []
    log["avg_params_p2"] = []
    log["params_p1"] = []
    log["params_p2"] = []
    # M is [P(p1 chooses action1 and p2 chooses action1), P(p1 chooses action2 and p2 chooses action1), etc.] or [a1a1, a2a1, a1a2, a2a2] for the two players.
    log["avg_Ms"] = []
    log["Ms"] = []
    log["avg_rewards_p1"] = []
    log["avg_rewards_p2"] = []
    log["rewards_p1"] = []
    log["rewards_p2"] = []
    log["seed"] = seed

    env = NonMfosMetaGames(
        b=batch_size,
        pms=pms,
        p1=p1,
        p2=p2,
        lr=lr,
        asym=asym,
        threshold=threshold,
        pwlinear=pwlinear,
        seed=seed,
        ccdr=ccdr,
        adam=adam,
    )
    env.reset()
    running_rew_p1 = torch.zeros(batch_size).to(device)
    running_rew_p2 = torch.zeros(batch_size).to(device)
    for i in range(num_steps):
        probs, r_p1, r_p2, M = env.step()
        running_rew_p1 += r_p1.squeeze(-1)
        running_rew_p2 += r_p2.squeeze(-1)

        log["avg_params_p1"].append(probs[0].detach().mean(dim=0).squeeze().tolist())
        log["avg_params_p2"].append(probs[1].detach().mean(dim=0).squeeze().tolist())
        log["params_p1"].append(
            probs[0].detach()[runs_to_track].squeeze().numpy().tolist()
        )
        log["params_p2"].append(
            probs[1].detach()[runs_to_track].squeeze().numpy().tolist()
        )
        log["avg_Ms"].append(M.detach().mean(dim=0).squeeze().tolist())
        log["Ms"].append(M.detach()[runs_to_track].squeeze().numpy().tolist())
        log["avg_rewards_p1"].append(r_p1.detach().mean(dim=0).squeeze().tolist())
        log["avg_rewards_p2"].append(r_p2.detach().mean(dim=0).squeeze().tolist())
        log["rewards_p1"].append(r_p1.detach()[runs_to_track].squeeze().tolist())
        log["rewards_p2"].append(r_p2.detach()[runs_to_track].squeeze().tolist())

    mean_rew_p1 = (running_rew_p1.mean() / num_steps).item()
    mean_rew_p2 = (running_rew_p2.mean() / num_steps).item()

    log["mean_rew_p1"] = mean_rew_p1
    log["mean_rew_p2"] = mean_rew_p2

    return log


def run_simulation(params):
    (
        pms,
        p1,
        p2,
        lr,
        asym,
        threshold,
        pwlinear,
        ccdr,
        adam,
        num_steps,
        batch_size,
        runs_to_track,
        seed,
    ) = params
    return get_log(
        pms,
        p1,
        p2,
        lr,
        asym,
        threshold,
        pwlinear,
        ccdr,
        adam,
        num_steps,
        batch_size,
        runs_to_track,
        seed,
    )


def get_params_tuple(log):
    return (
        log["p1"],
        log["p2"],
        log["lr"],
        log["asym"],
        log["threshold"],
        log["pwlinear"],
        log["ccdr"],
        log["adam"],
        log["num_steps"],
        log["batch_size"],
    )


if __name__ == "__main__":
    batch_size = 8192
    n_runs_to_track = 20
    num_steps = 500  # 100
    name = f"runs/{args.exp_name}"
    G = args.G

    # prisoner's dilemma
    pd_payoff_mat_1 = torch.Tensor([[G, 0], [1 + G, 1]]).to(device)
    pd_payoff_mat_2 = pd_payoff_mat_1.T
    pd = (pd_payoff_mat_1, pd_payoff_mat_2)
    pds = [pd]

    lrs = np.logspace(-0, 1.5, num=8)
    asyms = [None]
    # thresholds = [None, "abs", "exp0", "exp1", "squared", "quartic"]
    thresholds = [args.threshold]
    pwlinears = [args.pwlinear]
    ccdrs = [None]
    adams = [False]

    player_combos = [
        ("NL", "NL"),
        ("NL", "LOLA"),
        ("LOLA", "LOLA"),
    ]
    seeds = [42]

    assert n_runs_to_track <= batch_size

    cmd_line_args = {}
    cmd_line_args["batch_size"] = batch_size
    cmd_line_args["n_runs_to_track"] = n_runs_to_track
    cmd_line_args["num_steps"] = num_steps
    cmd_line_args["G"] = G
    cmd_line_args["threshold"] = thresholds
    cmd_line_args["pwlinear"] = pwlinears
    cmd_line_args["ccdr"] = ccdrs
    cmd_line_args["adam"] = adams
    cmd_line_args["pds"] = pds
    cmd_line_args["lrs"] = lrs
    cmd_line_args["asyms"] = asyms
    cmd_line_args["player_combos"] = player_combos
    cmd_line_args["seeds"] = seeds
    cmd_line_args["name"] = name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(cmd_line_args, f, indent=2)

    results = []
    param_list = []
    for pms in pds:
        for p1, p2 in player_combos:
            for lr in lrs:
                for asym in asyms:
                    for threshold in thresholds:
                        for pwlinear in pwlinears:
                            for ccdr in ccdrs:
                                for adam in adams:
                                    for seed in seeds:
                                        if seed is not None:
                                            np.random.seed(seed)
                                        runs_to_track = np.random.choice(
                                            batch_size, n_runs_to_track, replace=False
                                        )
                                        param_list.append(
                                            (
                                                pms,
                                                p1,
                                                p2,
                                                lr,
                                                asym,
                                                threshold,
                                                pwlinear,
                                                ccdr,
                                                adam,
                                                num_steps,
                                                batch_size,
                                                runs_to_track,
                                                seed,
                                            )
                                        )

    with Pool() as pool:
        start_time = time.time()
        results = pool.map(run_simulation, param_list)
        print(f"Elapsed time: {time.time() - start_time}")

    log_dict = {get_params_tuple(log): log for log in results}
    ordered_results = [log_dict[tuple(params[1:11])] for params in param_list]

    with open(os.path.join(name, f"out.json"), "w") as f:
        json.dump(ordered_results, f)

"""
TODO: 
1. Add logging and plotting code to the existing code.
    - Plotting:
        - Plot the ESV vs. training episode for each game.
        - Plot the average ESV (averaging over steps) vs. learning rate for each game. 
        
"""
