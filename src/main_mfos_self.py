import torch
from ppo import PPO, Memory
from environments import MetaGames, SymmetricMetaGames
import os
import argparse
import json
import numpy as np
from tqdm import tqdm

from plot_bigtime import main as plot_all


parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--tracing", action="store_true", default=False)
args = parser.parse_args()


if __name__ == "__main__":
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.0002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 256 # 1024 -> 256
    batch_size = 1024 # 4096 -> 1024
    random_seed = None
    num_steps = 100

    save_freq = 100 # 1000

    lamb = 1.0
    if not args.tracing: lamb = -1.0 # never the NL opponent. jank, but seems like the minimal change needed
    lamb_anneal = 0.0015
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(f"runs/{name}"):
        os.mkdir(f"runs/{name}")
        with open(os.path.join('runs', name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")

    #############################################

    # creating environment
    env = SymmetricMetaGames(batch_size, game=args.game)

    action_dim = env.d
    state_dim = env.d * 2

    memory_0 = Memory()
    memory_1 = Memory()

    ppo_0 = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, entropy=args.entropy)
    ppo_1 = PPO(state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, entropy=args.entropy)

    nl_env = MetaGames(batch_size, game=args.game)

    print(lr, betas)
    # training loop
    rew_means = []
    quartile_dumps = []

    for i_episode in range(1, max_episodes + 1):
        print("=" * 100)
        print(i_episode)
        print(lamb, flush=True)
        if lamb > 0:
            lamb -= lamb_anneal

        if i_episode in [1, max_episodes//4, max_episodes//2, 3*max_episodes//4, max_episodes]:
            print("v opponent for quartile dump") 
            state_0, state_1 = env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_reward_1 = torch.zeros(batch_size).to(device)

            all_params_1 = []
            all_Ms = []
            rewards_1 = []
            rewards_2 = []
            for t in tqdm(range(num_steps)):

                # Running policy_old:
                action_0 = ppo_0.policy_old.act(state_0, memory_0) # actions are "raw"/unsigmoided params
                action_1 = ppo_1.policy_old.act(state_1, memory_1)
                states, rewards, M = env.step(action_0, action_1)
                state_0, state_1 = states 
                # state_0 is like torch.cat((p_ba_0.detach(), p_ba_1.detach()), dim=-1)
                reward_0, reward_1 = rewards
                # rewards is l1, l2 = self.game_batched([p_ba_0, p_ba_1])
                # from below, l1 is for MFOS 1, l2 is for MFOS 0
                # this means that p_ba_1 is action_1, which is MFOS 0
                # and p_ba_0 is action_0, which is MFOS 1

                memory_0.rewards.append(reward_0)
                memory_1.rewards.append(reward_1)

                running_reward_0 += reward_0.squeeze(-1)
                running_reward_1 += reward_1.squeeze(-1)

                # print(state_0.shape) # torch.Size([4096, 10]) = torch.sigmoid(torch.cat((p_ba_0.detach(), p_ba_1.detach()), dim=-1))

                # if M is not None: # M was None for one-shot games, then arunim changed it
                M_1 = M[0, :].detach().tolist() # just the first run of the batch
                # alternatively, take the mean of the batch
                # M_1 = M.mean(dim=0).squeeze().tolist()
                all_Ms.append(M_1)

                # =======
                # # params 1 is supposed to recover MFOS 1's params aka p_ba_0
                # if args.game.find("I") != -1: # iterated games
                #     # params_1 = state[:batch_size, :] # has size batch_size x 10, so we need to split it
                #     params_1 = torch.split(state_0, [5, 5], dim=-1)[0] # second half of state = opponent or self idk idc its symmetric?
                #     # print(params_1.shape) # torch.Size([4096, 5])
                # else: # proxy for oneshot games
                #     params_1 = torch.split(state_0, [1, 1], dim=-1)[0] 

                # all_params_1.append(params_1[0,:].detach().tolist()) # just the first run of the batch
                # =======
                
                # alternatively, take the mean of the batch
                # all_params_1.append(params_1.mean(dim=0).squeeze().tolist())
                rewards_2.append(reward_0.mean(dim=0).squeeze().tolist()) # corresponds to MFOS 1
                rewards_1.append(reward_1.mean(dim=0).squeeze().tolist()) # corresponds to MFOS 0
            
            # =======
            # if args.game.find("I") != -1: # iterated games
            #     split_params = torch.split(state_0, [5, 5], dim=-1)
            # else: 
            #     split_params = torch.split(state_0, [1, 1], dim=-1)
            # end_params = [split_params[1][:50].tolist(), split_params[0][:50].tolist()]
            # =======

            quartile_dumps.append({
                "game": args.game,
                "opponent": "self",
                "timestep": i_episode,
                # "all_params_1": all_params_1,
                # "end_params": end_params,
                "all_Ms": all_Ms,
                "rewards_1": rewards_1,
                "rewards_2": rewards_2,
            })

            ppo_0.update(memory_0)
            ppo_1.update(memory_1)

            memory_0.clear_memory()
            memory_1.clear_memory()

            l0 = -running_reward_0.mean() / num_steps
            l1 = -running_reward_1.mean() / num_steps
            print(f"loss 0: {l0}")
            print(f"loss 1: {l1}")
            print(f"sum: {l0 + l1}")

            rew_means.append(
                {
                    "ep": i_episode,
                    "other": True,
                    "rew 0": -l0.item(),
                    "rew 1": -l1.item(),
                }
            )

        elif np.random.random() > lamb:
            print("v opponent")
            state_0, state_1 = env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_reward_1 = torch.zeros(batch_size).to(device)

            for t in tqdm(range(num_steps)):

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
                
            ppo_0.update(memory_0)
            ppo_1.update(memory_1)

            memory_0.clear_memory()
            memory_1.clear_memory()

            l0 = -running_reward_0.mean() / num_steps
            l1 = -running_reward_1.mean() / num_steps
            print(f"loss 0: {l0}")
            print(f"loss 1: {l1}")
            print(f"sum: {l0 + l1}")

            rew_means.append(
                {
                    "ep": i_episode,
                    "other": True,
                    "rew 0": -l0.item(),
                    "rew 1": -l1.item(),
                }
            )

        else:
            print("v nl")
            state = nl_env.reset()

            running_reward_0 = torch.zeros(batch_size).to(device)
            running_opp_reward = torch.zeros(batch_size).to(device)
            
            for t in range(num_steps):
                # Running policy_old:
                action = ppo_0.policy_old.act(state, memory_0)
                state, reward, info, M = nl_env.step(action)

                memory_0.rewards.append(reward)
                running_reward_0 += reward.squeeze(-1)
                running_opp_reward += info.squeeze(-1)

            ppo_0.update(memory_0)
            memory_0.clear_memory()

            print(f"loss 0: {-running_reward_0.mean() / num_steps}")
            print(f"opponent loss 0: {-running_opp_reward.mean() / num_steps}")

            # SECOND AGENT UPDATE
            state = nl_env.reset()

            running_reward_1 = torch.zeros(batch_size).to(device)
            running_opp_reward = torch.zeros(batch_size).to(device)

            for t in range(num_steps):
                # Running policy_old:
                action = ppo_1.policy_old.act(state, memory_1)
                state, reward, info, M = nl_env.step(action)

                memory_1.rewards.append(reward)
                running_reward_1 += reward.squeeze(-1)
                running_opp_reward += info.squeeze(-1)

            ppo_1.update(memory_1)
            memory_1.clear_memory()

            print(f"loss 1: {-running_reward_1.mean() / num_steps}")
            print(f"opponent loss 1: {-running_opp_reward.mean() / num_steps}")

            rew_means.append(
                {
                    "ep": i_episode,
                    "other": False,
                    "rew 0": (running_reward_0.mean() / num_steps).item(),
                    "rew 1": (running_reward_1.mean() / num_steps).item(),
                }
            )

        if i_episode % save_freq == 0:
            ppo_0.save(os.path.join("runs", name, f"{i_episode}_0.pth"))
            ppo_1.save(os.path.join("runs", name, f"{i_episode}_1.pth"))
            with open(os.path.join("runs", name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")

    ppo_0.save(os.path.join("runs", name, f"{i_episode}_0.pth"))
    ppo_1.save(os.path.join("runs", name, f"{i_episode}_1.pth"))
    with open(os.path.join("runs", name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    with open(os.path.join("runs", name, f"quartile_dumps.json"), "w") as f:
        json.dump(quartile_dumps, f)

    print(f"SAVING! {i_episode}")

    print(f"Running plot_bigtime on {name}")
    plot_all(name, caller="self", game=args.game)