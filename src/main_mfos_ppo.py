import torch
from ppo import PPO, Memory
from environments import MetaGames
import os
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")  # type: ignore


if __name__ == "__main__":
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 1024
    batch_size = 4096
    n_runs_to_track = 20
    num_steps = 100
    G = 3

    save_freq = 250
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(name):
        os.mkdir(name)
        with open(os.path.join(name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #############################################

    pd_payoff_mat_1 = torch.Tensor([[G, 0], [1 + G, 1]]).to(device)
    pd_payoff_mat_2 = pd_payoff_mat_1.T
    pd = (pd_payoff_mat_1, pd_payoff_mat_2)
    pds = [pd]

    # creating environment
    env = MetaGames(
        batch_size,
        pds[0],
        opponent="NL",
        lr=lr,
        asym=None,
        threshold=None,
        pwlinear=None,
        seed=42,
        ccdr=None,
        adam=True,
    )

    action_dim = env.d
    state_dim = env.d * 2

    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, args.entropy)

    if args.checkpoint:
        ppo.load(args.checkpoint)

    print(lr, betas)

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

        for t in range(num_steps):
            state = torch.cat((state[0], state[1]), dim=-1)

            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, info, M = env.step(action)

            memory.rewards.append(reward)
            running_reward += reward.squeeze(-1)
            running_opp_reward += info.squeeze(-1)
            last_reward = reward.squeeze(-1)

        ppo.update(memory)
        memory.clear_memory()

        print("=" * 100, flush=True)

        print(f"episode: {i_episode}", flush=True)

        print(f"loss: {-running_reward.mean() / num_steps}", flush=True)

        rew_means.append(
            {
                "rew": (running_reward.mean() / num_steps).item(),
                "opp_rew": (running_opp_reward.mean() / num_steps).item(),
            }
        )

        print(f"opponent loss: {-running_opp_reward.mean() / num_steps}", flush=True)

        if i_episode % save_freq == 0:
            ppo.save(os.path.join(name, f"{i_episode}.pth"))
            with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")

    ppo.save(os.path.join(name, f"{i_episode}.pth"))
    with open(os.path.join(name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    print(f"SAVING! {i_episode}")
