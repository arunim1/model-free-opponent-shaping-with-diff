import torch
from ppo import PPO, Memory
from environments import MetaGames
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import time
from torch.multiprocessing import Pool, Process, set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass


parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, default="delete")
parser.add_argument("--G", type=float, default=2)
parser.add_argument("--threshold", type=str, default=None)
parser.add_argument("--pwlinear", type=int, default=None)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--checkpoint", type=str, default="")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")  # type: ignore


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
    log["five_game_logs"] = []
    log["seed"] = seed

    env = MetaGames(
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

    action_dim = env.d
    state_dim = env.d * 2

    memory = Memory()
    ppo = PPO(
        state_dim, action_dim, mfos_lr, betas, gamma, K_epochs, eps_clip, args.entropy
    )

    if args.checkpoint:
        ppo.load(args.checkpoint)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    rew_means = []

    for i_episode in range(1, max_episodes + 1):
        state = env.reset()

        running_reward = torch.zeros(batch_size).to(device)
        running_opp_reward = torch.zeros(batch_size).to(device)

        last_reward = 0

        if i_episode in [
            1,
            max_episodes // 4,
            max_episodes // 2,
            3 * max_episodes // 4,
            max_episodes,
        ]:
            sub_log = {}
            sub_log["avg_params_p1"] = []
            sub_log["avg_params_p2"] = []
            sub_log["params_p1"] = []
            sub_log["params_p2"] = []
            # M is [P(p1 chooses action1 and p2 chooses action1), P(p1 chooses action2 and p2 chooses action1), etc.] or [a1a1, a2a1, a1a2, a2a2] for the two players.
            sub_log["avg_Ms"] = []
            sub_log["Ms"] = []
            sub_log["avg_rewards_p1"] = []
            sub_log["avg_rewards_p2"] = []
            sub_log["rewards_p1"] = []
            sub_log["rewards_p2"] = []

        for t in range(num_steps):
            state = torch.cat((state[0], state[1]), dim=-1)

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, opp_reward, M = env.step(action)

            memory.rewards.append(reward)
            running_reward += reward.squeeze(-1)
            running_opp_reward += opp_reward.squeeze(-1)
            last_reward = reward.squeeze(-1)

            if i_episode in [
                1,
                max_episodes // 4,
                max_episodes // 2,
                3 * max_episodes // 4,
                max_episodes,
            ]:
                sub_log["avg_params_p1"].append(
                    state[0].detach().mean(dim=0).squeeze().tolist()
                )
                sub_log["avg_params_p2"].append(
                    state[1].detach().mean(dim=0).squeeze().tolist()
                )
                sub_log["params_p1"].append(
                    state[0].detach()[runs_to_track].squeeze().numpy().tolist()
                )
                sub_log["params_p2"].append(
                    state[1].detach()[runs_to_track].squeeze().numpy().tolist()
                )
                sub_log["avg_Ms"].append(M.detach().mean(dim=0).squeeze().tolist())
                sub_log["Ms"].append(
                    M.detach()[runs_to_track].squeeze().numpy().tolist()
                )
                sub_log["avg_rewards_p1"].append(reward.mean(dim=0).squeeze().tolist())
                sub_log["avg_rewards_p2"].append(
                    opp_reward.mean(dim=0).squeeze().tolist()
                )
                sub_log["rewards_p1"].append(
                    reward.detach()[runs_to_track].squeeze().tolist()
                )
                sub_log["rewards_p2"].append(
                    opp_reward.detach()[runs_to_track].squeeze().tolist()
                )

        if i_episode in [
            1,
            max_episodes // 4,
            max_episodes // 2,
            3 * max_episodes // 4,
            max_episodes,
        ]:
            log["five_game_logs"].append(sub_log)

        ppo.update(memory)
        memory.clear_memory()

        rew_means.append(
            {
                "rew": (running_reward.mean() / num_steps).item(),
                "opp_rew": (running_opp_reward.mean() / num_steps).item(),
            }
        )

        # if i_episode % save_freq == 0:
        #     ppo.save(os.path.join(name, f"{i_episode}.pth"))
        #     with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
        #         json.dump(rew_means, f)
        #     print(f"SAVING! {i_episode}")

    log["rew_means"] = rew_means

    # ppo.save(os.path.join(name, f"{i_episode}.pth"))

    return log


def run_simulation(params):
    (
        pms,
        opponent,
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
        "MFOS",
        opponent,
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
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    mfos_lr = 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 256  # 1024
    batch_size = 4096
    n_runs_to_track = 20
    num_steps = 500
    G = args.G

    save_freq = max_episodes // 4  # 250
    name = f"runs/{args.exp_name}"

    #############################################

    pd_payoff_mat_1 = torch.Tensor([[G, 0], [1 + G, 1]]).to(device)
    pd_payoff_mat_2 = pd_payoff_mat_1.T
    pd = (pd_payoff_mat_1, pd_payoff_mat_2)
    pds = [pd]
    lrs = [0.1, 0.3, 1, 3, 10]
    asyms = [None]
    thresholds = [args.threshold]
    pwlinears = [args.pwlinear]
    ccdrs = [None]
    adams = [False]

    opponents = ["NL", "LOLA"]

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
    cmd_line_args["opponents"] = opponents
    cmd_line_args["seeds"] = seeds
    cmd_line_args["name"] = name
    cmd_line_args["K_epochs"] = K_epochs
    cmd_line_args["eps_clip"] = eps_clip
    cmd_line_args["gamma"] = gamma
    cmd_line_args["mfos_lr"] = mfos_lr
    cmd_line_args["betas"] = betas
    cmd_line_args["max_episodes"] = max_episodes

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "runs/commandline_args.txt"), "w") as f:
            json.dump(cmd_line_args, f, indent=2)

    results = []
    param_list = []
    for pms in pds:
        for opponent in opponents:
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
                                                opponent,
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

    with Pool(4) as pool:
        start_time = time.time()
        results = pool.map(run_simulation, param_list)
        print(f"Elapsed time: {time.time() - start_time}")
        if not os.path.isdir(f"runs/{name}"):
            os.mkdir(f"runs/{name}")

    log_dict = {get_params_tuple(log): log for log in results}
    ordered_results = [log_dict[tuple(params[1:11])] for params in param_list]

    with open(os.path.join(name, f"out.json"), "w") as f:
        json.dump(ordered_results, f)
