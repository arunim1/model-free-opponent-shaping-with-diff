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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")  # type: ignore


# TODO: Mfos self-play is not cooperating yet, by the looks of it.


def get_log(
    pms,
    p1,
    p2,
    lr,
    mfos_lr,
    betas,
    gamma,
    K_epochs,
    eps_clip,
    max_episodes,
    lamb,
    lamb_anneal,
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
    log = {}
    log["payoff_mat_p2"] = pms[0].cpu().numpy().tolist()
    log["payoff_mat_p1"] = pms[1].cpu().numpy().tolist()
    log["p1"] = p1
    log["p2"] = p2
    log["lr"] = lr
    log["mfos_lr"] = mfos_lr
    log["asym"] = asym
    log["threshold"] = threshold
    log["pwlinear"] = pwlinear
    log["ccdr"] = ccdr
    log["adam"] = adam
    log["num_steps"] = num_steps
    log["batch_size"] = batch_size
    log["seed"] = seed
    log["five_game_logs"] = []

    # creating environment
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
        mfos_lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        entropy=args.entropy,
    )
    ppo_1 = PPO(
        state_dim,
        action_dim,
        mfos_lr,
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

    for i_episode in range(1, max_episodes + 1):
        # print("=" * 100)
        # print(i_episode)
        # print(lamb, flush=True)
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

        if lamb > 0:
            lamb -= lamb_anneal
        if np.random.random() > lamb or i_episode in [
            1,
            max_episodes // 4,
            max_episodes // 2,
            3 * max_episodes // 4,
            max_episodes,
        ]:
            # print("v opponent")
            probs = env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_reward_1 = torch.zeros(batch_size).to(device)

            for t in range(num_steps):
                # Reconstruct state_0 and state_1
                state_0 = torch.cat((probs[1], probs[0]), dim=-1)
                state_1 = torch.cat((probs[0], probs[1]), dim=-1)

                # Running policy_old:
                action_0 = ppo_0.policy_old.act(state_0, memory_0)
                action_1 = ppo_1.policy_old.act(state_1, memory_1)
                probs, rewards, M = env.step(action_0, action_1)

                reward_0, reward_1 = rewards

                memory_0.rewards.append(reward_0)
                memory_1.rewards.append(reward_1)

                running_reward_0 += reward_0.squeeze(-1)
                running_reward_1 += reward_1.squeeze(-1)

                if i_episode in [
                    1,
                    max_episodes // 4,
                    max_episodes // 2,
                    3 * max_episodes // 4,
                    max_episodes,
                ]:
                    sub_log["avg_params_p1"].append(
                        probs[0].detach().mean(dim=0).squeeze().tolist()
                    )
                    sub_log["avg_params_p2"].append(
                        probs[1].detach().mean(dim=0).squeeze().tolist()
                    )
                    sub_log["params_p1"].append(
                        probs[0]
                        .detach()[runs_to_track]
                        .squeeze()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    sub_log["params_p2"].append(
                        probs[1]
                        .detach()[runs_to_track]
                        .squeeze()
                        .cpu()
                        .numpy()
                        .tolist()
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

            if i_episode in [
                1,
                max_episodes // 4,
                max_episodes // 2,
                3 * max_episodes // 4,
                max_episodes,
            ]:
                log["five_game_logs"].append(sub_log)

            ppo_0.update(memory_0)
            ppo_1.update(memory_1)

            memory_0.clear_memory()
            memory_1.clear_memory()

            l0 = -running_reward_0.mean() / num_steps
            l1 = -running_reward_1.mean() / num_steps
            # print(f"loss 0: {l0}")
            # print(f"loss 1: {l1}")
            # print(f"sum: {l0 + l1}")

            rew_means.append(
                {
                    "ep": i_episode,
                    "other": True,
                    "rew 0": -l0.item(),
                    "rew 1": -l1.item(),
                }
            )

        else:
            # print("v nl")
            state = nl_env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_opp_reward = torch.zeros(batch_size).to(device)

            for t in range(num_steps):
                state = torch.cat((state[0], state[1]), dim=-1)

                # Running policy_old:
                action = ppo_0.policy_old.act(state, memory_0)
                state, reward, opp_reward, M = nl_env.step(action)

                memory_0.rewards.append(reward)
                running_reward_0 += reward.squeeze(-1)
                running_opp_reward += opp_reward.squeeze(-1)

            ppo_0.update(memory_0)
            memory_0.clear_memory()

            # print(f"loss 0: {-running_reward_0.mean() / num_steps}")
            # print(f"opponent loss 0: {-running_opp_reward.mean() / num_steps}")

            # SECOND AGENT UPDATE
            state = nl_env.reset()

            running_reward_1 = torch.zeros(batch_size).to(device)
            running_opp_reward = torch.zeros(batch_size).to(device)

            for t in range(num_steps):
                state = torch.cat((state[0], state[1]), dim=-1)

                # Running policy_old:
                action = ppo_1.policy_old.act(state, memory_1)
                state, reward, opp_reward, M = nl_env.step(action)

                memory_1.rewards.append(reward)
                running_reward_1 += reward.squeeze(-1)
                running_opp_reward += opp_reward.squeeze(-1)

            ppo_1.update(memory_1)
            memory_1.clear_memory()

            # print(f"loss 1: {-running_reward_1.mean() / num_steps}")
            # print(f"opponent loss 1: {-running_opp_reward.mean() / num_steps}")

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
        mfos_lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        max_episodes,
        lamb,
        lamb_anneal,
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
        "MFOS 1",
        "MFOS 2",
        lr,
        mfos_lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        max_episodes,
        lamb,
        lamb_anneal,
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
        log["lr"],
        log["mfos_lr"],
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

    mfos_lr = 0.0002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 256  # 1024
    batch_size = 1024  # 4096
    n_runs_to_track = 20
    num_steps = 500  # 100
    G = args.G

    save_freq = max_episodes // 4  # 250

    lamb = -1.0
    lamb_anneal = 0.006  # 0.0015
    name = f"runs/{args.exp_name}"

    #############################################

    # ezc
    pd_payoff_mat_1 = torch.Tensor([[10.0, 3.0], [2.0, 0.0]]).to(device)
    pd_payoff_mat_2 = pd_payoff_mat_1.T
    pd = (pd_payoff_mat_1, pd_payoff_mat_2)
    pds = [pd]
    lrs = [0.1, 0.3, 1, 3, 10]
    asyms = [None]
    thresholds = [args.threshold]
    pwlinears = [args.pwlinear]
    ccdrs = [None]
    adams = [False]

    seeds = [42]

    assert n_runs_to_track <= batch_size

    cmd_line_args = {}
    cmd_line_args["batch_size"] = batch_size
    cmd_line_args["n_runs_to_track"] = 20
    cmd_line_args["num_steps"] = num_steps
    cmd_line_args["G"] = G
    cmd_line_args["threshold"] = thresholds
    cmd_line_args["pwlinear"] = pwlinears
    cmd_line_args["ccdr"] = ccdrs
    cmd_line_args["adam"] = adams
    cmd_line_args["payoff_mat_p1"] = pd_payoff_mat_1.cpu().numpy().tolist()
    cmd_line_args["lrs"] = lrs
    cmd_line_args["asyms"] = asyms
    cmd_line_args["seeds"] = seeds
    cmd_line_args["name"] = name
    cmd_line_args["K_epochs"] = K_epochs
    cmd_line_args["eps_clip"] = eps_clip
    cmd_line_args["gamma"] = gamma
    cmd_line_args["mfos_lr"] = mfos_lr
    cmd_line_args["betas"] = betas
    cmd_line_args["max_episodes"] = max_episodes
    cmd_line_args["save_freq"] = save_freq
    cmd_line_args["lamb"] = lamb
    cmd_line_args["lamb_anneal"] = lamb_anneal

    # print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(cmd_line_args, f, indent=2)

    results = []
    param_list = []
    for pms in pds:
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
                                            lr,
                                            mfos_lr,
                                            betas,
                                            gamma,
                                            K_epochs,
                                            eps_clip,
                                            max_episodes,
                                            lamb,
                                            lamb_anneal,
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

    with Pool(8) as pool:
        start_time = time.time()
        results = pool.map(run_simulation, param_list)
        print(f"Elapsed time: {time.time() - start_time}")

    log_dict = {get_params_tuple(log): log for log in results}
    ordered_results = []
    for params in param_list:
        tup = tuple(params[1:3] + params[10:17])
        ordered_results.append(log_dict[tup])

    with open(os.path.join(name, f"out.json"), "w") as f:
        json.dump(ordered_results, f)
