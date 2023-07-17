import torch
from ppo import PPO, Memory
from environments import MetaGames
import os
import argparse
import json
from plot_bigtime import main as plot_all

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--mamaml-id", type=int, default=0)
args = parser.parse_args()


if __name__ == "__main__":
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    max_episodes = 256 # 1024
    batch_size = 1024 # 4096
    random_seed = None
    num_steps = 100

    save_freq = 250
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(f"runs/{name}"):
        os.mkdir(f"runs/{name}")
        with open(os.path.join('runs', name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    #############################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu")

    # creating environment
    env = MetaGames(batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id)

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

    quartile_dumps = []

    for i_episode in range(1, max_episodes + 1): # outer. 
        state = env.reset()

        running_reward = torch.zeros(batch_size).to(device)
        running_opp_reward = torch.zeros(batch_size).to(device)

        last_reward = 0

        if i_episode in [1, max_episodes//4, max_episodes//2, 3*max_episodes//4, max_episodes]:
            
            all_params_1 = []
            all_Ms = []
            rewards_1 = []
            rewards_2 = []

            for t in range(num_steps): # inner

                # Running policy_old:
                action = ppo.policy_old.act(state, memory)
                state, reward, info, M = env.step(action) # actually more like params, r0, r1, M

                memory.rewards.append(reward)
                running_reward += reward.squeeze(-1)
                running_opp_reward += info.squeeze(-1)
                last_reward = reward.squeeze(-1)

                M_1 = M[0, :].detach().tolist() # just the first run of the batch
                # alternatively, take the mean of the batch
                # M_1 = M.mean(dim=0).squeeze().tolist()
                all_Ms.append(M_1)
                
                if args.game.find("I") != -1: # iterated games 
                    # params_1 = state[:batch_size, :] # has size batch_size x 10, so we need to split it
                    params_1 = torch.split(state, [5, 5], dim=-1)[1] # second half of state = opponent
                else: # proxy for oneshot games
                    params_1 = torch.split(state, [1, 1], dim=-1)[1] 

                all_params_1.append(params_1[0,:].detach().tolist()) # just the first run of the batch
                # alternatively, take the mean of the batch
                # all_params_1.append(params_1.mean(dim=0).squeeze().tolist())
                rewards_1.append(reward.mean(dim=0).squeeze().tolist()) # MFOS reward
                rewards_2.append(info.mean(dim=0).squeeze().tolist()) # opponent reward
            
            if args.game.find("I") != -1: # iterated games
                split_params = torch.split(state, [5, 5], dim=-1)
            else: 
                split_params = torch.split(state, [1, 1], dim=-1)
            end_params = [split_params[0][:50].tolist(), split_params[1][:50].tolist()]
            
            quartile_dumps.append({
                "game": args.game,
                "opponent": args.opponent,
                "timestep": i_episode,
                "all_params_1": all_params_1,
                "end_params": end_params,
                "all_Ms": all_Ms,
                "rewards_1": rewards_1,
                "rewards_2": rewards_2,
            })
        else: 
            for t in range(num_steps): # inner

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

        rew_means.append( # this is what is dumped to the json file
            {
                "rew": (running_reward.mean() / num_steps).item(),
                "opp_rew": (running_opp_reward.mean() / num_steps).item(),
            }
        )

        print(f"opponent loss: {-running_opp_reward.mean() / num_steps}", flush=True)

        if i_episode % save_freq == 0:
            ppo.save(os.path.join("runs", name, f"{i_episode}.pth"))
            with open(os.path.join("runs", name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")

    ppo.save(os.path.join("runs", name, f"{i_episode}.pth"))
    with open(os.path.join("runs", name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    with open(os.path.join("runs", name, f"quartile_dumps.json"), "w") as f:
        json.dump(quartile_dumps, f)

    print(f"SAVING! {i_episode}")

    print("Running plot_bigtime.py with filename: ", name)
    
    plot_all(name, caller="ppo", opponent=args.opponent, game=args.game)