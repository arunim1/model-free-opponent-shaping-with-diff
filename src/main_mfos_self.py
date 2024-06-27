import torch
from ppo import PPO, Memory

from environments import MetaGames, SymmetricMetaGames
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
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--G", type=float, default=2)
parser.add_argument("--threshold", type=str, default=None)
parser.add_argument("--pwlinear", type=int, default=None)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--checkpoint", type=str, default="")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # type: ignore


def get_log(
    pms,
    p1,
    p2,
    lr,
    mfos_lr1,
    mfos_lr2,
    betas,
    gamma,
    K_epochs,
    eps_clip,
    max_episodes,
    asym,
    threshold,
    pwlinear,
    ccdr,
    adam,
    num_steps,
    batch_size,
    runs_to_track,
    seed,
    lamb,
    lamb_anneal,
):
    # initialize empty log for this game
    log = {}
    log["payoff_mat_p2"] = pms[0].cpu().numpy().tolist()
    log["payoff_mat_p1"] = pms[1].cpu().numpy().tolist()
    log["p1"] = p1
    log["p2"] = p2
    log["lr"] = lr
    log["mfos_lr1"] = mfos_lr1
    log["mfos_lr2"] = mfos_lr2
    log["asym"] = asym
    log["threshold"] = threshold
    log["pwlinear"] = pwlinear
    log["ccdr"] = ccdr
    log["adam"] = adam
    log["num_steps"] = num_steps
    log["batch_size"] = batch_size
    log["seed"] = seed
    log["five_game_logs"] = []

    env = SymmetricMetaGames(
        batch_size,
        pms=pms,
        asym=asym,
        threshold=threshold,
        pwlinear=pwlinear,
        seed=seed,
        ccdr=ccdr,
    )

    action_dim = env.d
    state_dim = env.d * 2

    memory_0 = Memory()
    memory_1 = Memory()

    ppo_0 = PPO(
        state_dim,
        action_dim,
        mfos_lr1,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        entropy=args.entropy,
    )
    ppo_1 = PPO(
        state_dim,
        action_dim,
        mfos_lr2,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        entropy=args.entropy,
    )

    nl_env = MetaGames(
        batch_size,
        pms=pms,
        opponent="NL",
        lr=lr,
        asym=asym,
        threshold=threshold,
        pwlinear=pwlinear,
        seed=seed,
        ccdr=ccdr,
        adam=adam,
    )

    # training loop
    rew_means = []

    for i_episode in tqdm(range(1, max_episodes + 1)):
        subtime = i_episode in [
            1,
            max_episodes // 4,
            max_episodes // 2,
            3 * max_episodes // 4,
            max_episodes,
        ]
        if subtime:
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

        if lamb > 0:
            lamb -= lamb_anneal
        if (np.random.random() > lamb) or subtime:
            state_0, state_1 = env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_reward_1 = torch.zeros(batch_size).to(device)

            for t in range(num_steps):
                # Running policy_old:
                action_0 = ppo_0.policy_old.act(state_0, memory_0)
                action_1 = ppo_1.policy_old.act(state_1, memory_1)
                states, rewards, M = env.step(action_0, action_1)
                state_0, state_1 = states
                reward_0, reward_1 = rewards

                memory_0.rewards.append(reward_0)
                memory_1.rewards.append(reward_1)

                running_reward_0 += reward_0.squeeze(-1)
                running_reward_1 += reward_1.squeeze(-1)

                if subtime:
                    d_act = state_0.shape[1] // 2
                    p_ba_0, p_ba_1 = torch.split(state_0, d_act, dim=1)

                    # p_ba_0 corresponds to l1 aka reward_0 aka ppo_0 and p1. pleased
                    sub_log["avg_params_p1"].append(
                        p_ba_0.detach().mean(dim=0).squeeze().tolist()
                    )
                    sub_log["avg_params_p2"].append(
                        p_ba_1.detach().mean(dim=0).squeeze().tolist()
                    )
                    sub_log["params_p1"].append(
                        p_ba_0.detach()[runs_to_track].squeeze().cpu().numpy().tolist()
                    )
                    sub_log["params_p2"].append(
                        p_ba_1.detach()[runs_to_track].squeeze().cpu().numpy().tolist()
                    )
                    sub_log["avg_Ms"].append(
                        M.detach().mean(dim=0).squeeze().cpu().tolist()
                    )
                    sub_log["Ms"].append(
                        M.detach()[runs_to_track].squeeze().cpu().numpy().tolist()
                    )
                    sub_log["avg_rewards_p1"].append(
                        reward_0.mean(dim=0).squeeze().cpu().tolist()
                    )
                    sub_log["avg_rewards_p2"].append(
                        reward_1.mean(dim=0).squeeze().cpu().tolist()
                    )
                    sub_log["rewards_p1"].append(
                        reward_0.detach()[runs_to_track].squeeze().cpu().tolist()
                    )
                    sub_log["rewards_p2"].append(
                        reward_1.detach()[runs_to_track].squeeze().cpu().tolist()
                    )

            if subtime:
                log["five_game_logs"].append(sub_log)

            ppo_0.update(memory_0)
            ppo_1.update(memory_1)

            memory_0.clear_memory()
            memory_1.clear_memory()

            l0 = -running_reward_0.mean() / num_steps
            l1 = -running_reward_1.mean() / num_steps

            rew_means.append(
                {
                    "ep": i_episode,
                    "other": True,
                    "rew 0": -l0.item(),
                    "rew 1": -l1.item(),
                }
            )

        else:
            state = nl_env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_opp_reward = torch.zeros(batch_size).to(device)

            for t in range(num_steps):
                state = torch.cat((state[0], state[1]), dim=-1)

                # Running policy_old:
                action = ppo_0.policy_old.act(state, memory_0)
                state, reward, info, M = nl_env.step(action)

                memory_0.rewards.append(reward)
                running_reward_0 += reward.squeeze(-1)
                running_opp_reward += info.squeeze(-1)

            ppo_0.update(memory_0)
            memory_0.clear_memory()

            # SECOND AGENT UPDATE
            state = nl_env.reset()

            running_reward_1 = torch.zeros(batch_size).to(device)
            running_opp_reward = torch.zeros(batch_size).to(device)

            for t in range(num_steps):
                state = torch.cat((state[0], state[1]), dim=-1)

                # Running policy_old:
                action = ppo_1.policy_old.act(state, memory_1)
                state, reward, info, M = nl_env.step(action)

                memory_1.rewards.append(reward)
                running_reward_1 += reward.squeeze(-1)
                running_opp_reward += info.squeeze(-1)

            ppo_1.update(memory_1)
            memory_1.clear_memory()

            rew_means.append(
                {
                    "ep": i_episode,
                    "other": False,
                    "rew 0": (running_reward_0.mean() / num_steps).item(),
                    "rew 1": (running_reward_1.mean() / num_steps).item(),
                }
            )

    log["rew_means"] = rew_means

    return log


def run_simulation(params):
    (
        pms,
        lr,
        mfos_lr1,
        mfos_lr2,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        max_episodes,
        asym,
        threshold,
        pwlinear,
        ccdr,
        adam,
        num_steps,
        batch_size,
        runs_to_track,
        seed,
        lamb,
        lamb_anneal,
    ) = params
    return get_log(
        pms,
        "MFOS 1",
        "MFOS 2",
        lr,
        mfos_lr1,
        mfos_lr2,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        max_episodes,
        asym,
        threshold,
        pwlinear,
        ccdr,
        adam,
        num_steps,
        batch_size,
        runs_to_track,
        seed,
        lamb,
        lamb_anneal,
    )


def get_params_tuple(log):
    return (
        log["p1"],
        log["lr"],
        log["mfos_lr1"],
        log["mfos_lr2"],
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

    lr = 0.0002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 256
    batch_size = 256
    n_runs_to_track = 20
    num_steps = 250
    G = args.G

    save_freq = max_episodes // 4  # 250

    lamb = 1.0
    lamb_anneal = 0.0015
    name = f"runs/self/{args.exp_name}"

    # print(f"RUNNING NAME: {name}")
    # if not os.path.isdir(name):
    #     os.mkdir(name)
    #     with open(os.path.join(name, "commandline_args.txt"), "w") as f:
    #         json.dump(args.__dict__, f, indent=2)

    #############################################

    # creating environment
    pd_payoff_mat_1 = torch.Tensor([[3, 0], [1 + 3, 1]]).to(device)
    pd_payoff_mat_2 = pd_payoff_mat_1.T
    pd = (pd_payoff_mat_1, pd_payoff_mat_2)
    pds = [pd]
    lrs = [1]
    asyms = [None]
    thresholds = [args.threshold]
    ccdrs = [None]
    adams = [False]
    pwlinears = [args.pwlinear]
    seeds = [42]

    mfos1_lrs = [0.002]
    mfos2_lrs = [0.002]
    opponents = ["NL"]

    assert n_runs_to_track <= batch_size

    run_args = {}
    run_args["batch_size"] = batch_size
    run_args["n_runs_to_track"] = n_runs_to_track
    run_args["num_steps"] = num_steps
    run_args["G"] = G
    run_args["threshold"] = thresholds
    run_args["pwlinear"] = pwlinears
    run_args["ccdr"] = ccdrs
    run_args["adam"] = adams
    run_args["payoff_mat_p1"] = pd_payoff_mat_1.cpu().numpy().tolist()
    run_args["payoff_mat_p2"] = pd_payoff_mat_2.cpu().numpy().tolist()
    run_args["lrs"] = lrs
    run_args["asyms"] = asyms
    run_args["opponents"] = opponents
    run_args["seeds"] = seeds
    run_args["name"] = name
    run_args["K_epochs"] = K_epochs
    run_args["eps_clip"] = eps_clip
    run_args["gamma"] = gamma
    run_args["mfos1_lrs"] = mfos1_lrs
    run_args["mfos2_lrs"] = mfos2_lrs
    run_args["betas"] = betas
    run_args["max_episodes"] = max_episodes
    run_args["save_freq"] = save_freq
    run_args["lamb"] = lamb
    run_args["lamb_anneal"] = lamb_anneal

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "run_args.txt"), "w") as f:
            json.dump(run_args, f, indent=2)

    results = []
    param_list = []
    for pms in pds:
        for lr in lrs:
            for mfos_lr1 in mfos1_lrs:
                for mfos_lr2 in mfos2_lrs:
                    for asym in asyms:
                        for threshold in thresholds:
                            for pwlinear in pwlinears:
                                for ccdr in ccdrs:
                                    for adam in adams:
                                        for seed in seeds:
                                            if seed is not None:
                                                np.random.seed(seed)
                                            runs_to_track = np.random.choice(
                                                batch_size,
                                                n_runs_to_track,
                                                replace=False,
                                            )
                                            param_list.append(
                                                (
                                                    pms,
                                                    lr,
                                                    mfos_lr1,
                                                    mfos_lr2,
                                                    betas,
                                                    gamma,
                                                    K_epochs,
                                                    eps_clip,
                                                    max_episodes,
                                                    asym,
                                                    threshold,
                                                    pwlinear,
                                                    ccdr,
                                                    adam,
                                                    num_steps,
                                                    batch_size,
                                                    runs_to_track,
                                                    seed,
                                                    lamb,
                                                    lamb_anneal,
                                                )
                                            )

    with Pool(8) as pool:
        start_time = time.time()
        results = pool.map(run_simulation, param_list)
        print(f"Elapsed time: {time.time() - start_time}")

    log_dict = {get_params_tuple(log): log for log in results}
    # ordered_results = [log_dict[tuple(params[1:11])] for params in param_list]
    ordered_results = []
    for params in param_list:
        tup = tuple(params[1:11] + params[13:17])
        ordered_results.append(log_dict[tup])

    with open(os.path.join(name, f"out.json"), "w") as f:
        json.dump(ordered_results, f)
