import torch
from ppo_clean import PPO, Memory
from environments import MetaGames
import os
import argparse
import json
from plot_bigtime import main as plot_all
from diff_graphs import main as diff_graphs

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, required=True)
parser.add_argument("--opponent", type=str, required=True)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--exp-name", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--mamaml-id", type=int, default=0)
parser.add_argument("--nn-game", action="store_true", default=False)
parser.add_argument("--ccdr", action="store_true", default=False)
parser.add_argument("--pwlinear", action="store_true", default=False)
args = parser.parse_args()


if __name__ == "__main__":
    ############################################
    K_epochs = 4  # update policy for K epochs

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.002 #0.002  # parameters for Adam optimizer
    betas = (0.9, 0.999)

    # max_episodes = 512 # 1024
    # batch_size = 256
    # random_seed = None
    # num_steps = 300 

    max_episodes = 256
    batch_size = 4096
    random_seed = None
    num_steps = 200

    save_freq = 250
    name = args.exp_name

    print(f"RUNNING NAME: {name}")
    if not os.path.isdir(f"runs/{name}"):
        os.mkdir(f"runs/{name}")
        with open(os.path.join('runs', name, "commandline_args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" if torch.backends.mps.is_available() else "cpu") # type: ignore
    nn_game = args.nn_game
    ccdr = args.ccdr 
    pwlinear = args.pwlinear
    #############################################

    # creating environment
    env = MetaGames(batch_size, opponent=args.opponent, game=args.game, mmapg_id=args.mamaml_id, nn_game=nn_game, pwlinear=pwlinear, ccdr=ccdr, n_neurons=10)

    action_dim = env.d
    state_dim = env.d * 2
    print("action dim:", action_dim) # 3565. this is because it sees each parameter update as a separate action? 
    print("state dim:", state_dim) # 7130 = 3565 * 2 params + opponent params

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

    i_episode = 0
    # using torch.profiler.profile
    # with torch.profiler.profile(profile_memory=True, record_shapes=True) as prof:

    for i_episode in range(1, max_episodes + 1): # outer.

        state = env.reset()

        running_reward = torch.zeros(batch_size).to(device)
        running_opp_reward = torch.zeros(batch_size).to(device)

        last_reward = 0
        M_means = []

        if i_episode in [1, max_episodes//4, max_episodes//2, 3*max_episodes//4, max_episodes]: # for testing/isolating where the memory usage is so high. 
        # if False:
            
            all_params_1 = []
            end_params = []
            all_Ms = []
            rewards_1 = []
            rewards_2 = []
            
            M_mean = torch.zeros(4).to(device)
            pbar = tqdm(range(num_steps), desc="Inner: ")
            for t in pbar: # inner

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
                M_mean += M.detach().mean(dim=0).squeeze() # has shape (4,)
                all_Ms.append(M_1)

                if not (nn_game or pwlinear):
                    if args.game.find("I") != -1: # iterated games 
                        # params_1 = state[:batch_size, :] # has size batch_size x 10, so we need to split it
                        params_1 = torch.split(state, [5, 5], dim=-1)[1] # second half of state = opponent
                    else: # proxy for oneshot games
                        params_1 = torch.split(state, [1, 1], dim=-1)[1] 

                    all_params_1.append(params_1[0,:].detach().tolist()) # just the first run of the batch

                rewards_1.append(reward.mean(dim=0).squeeze().tolist()) # MFOS reward
                rewards_2.append(info.mean(dim=0).squeeze().tolist()) # opponent rewardarc

                if t == num_steps - 1 and pwlinear: 
                    with torch.no_grad():
                        all_params = torch.split(state, [int(state.shape[1]/2), int(state.shape[1]/2)], dim=-1)
                        diff_graphs(all_params[0].detach().cpu(), all_params[1].detach().cpu(), iterated = args.game.find("I") != -1, p1="MFOS", p2=args.opponent, name=f"{name}/MFOS{args.opponent}_{args.game}_{ccdr}")

                pbar.set_description(f"r0: {reward.detach().mean().item():.2f}, r1: {info.detach().mean().item():.2f}")
            
            M_means = (M_mean / num_steps).tolist()
            
            if not (nn_game or pwlinear):
                if args.game.find("I") != -1: # iterated games
                    split_params = torch.split(state, [5, 5], dim=-1)
                else: 
                    split_params = torch.split(state, [1, 1], dim=-1)
                end_params = [split_params[0][:50].tolist(), split_params[1][:50].tolist()]
            
            quartile_dumps.append({
                "game": args.game,
                "opponent": args.opponent,
                "timestep": i_episode,
                "all_params_1": all_params_1 if not nn_game else None,
                "end_params": end_params if not nn_game else None,
                "all_Ms": all_Ms,
                "rewards_1": rewards_1,
                "rewards_2": rewards_2,
            })
        else: 
            # M_mean = torch.zeros(4).to(device)
            pbar = tqdm(range(num_steps), desc="Inner: ")
            for t in pbar: # inner
                # Running policy_old:
                action = ppo.policy_old.act(state, memory)
                state, reward, info, M = env.step(action)
                # M_mean += M.detach().mean(dim=0).squeeze() # has shape (4,)

                memory.rewards.append(reward)
                running_reward += reward.squeeze(-1)
                running_opp_reward += info.squeeze(-1)
                last_reward = reward.squeeze(-1)
                # pbar.set_description(f"r0: {r0.detach().mean().item():.2f}, r1: {r1.detach().mean().item():.2f}")
                pbar.set_description(f"r0: {reward.detach().mean().item():.2f}, r1: {info.detach().mean().item():.2f}")
            # M_means = (M_mean / num_steps).tolist()
    

        ppo.update(memory)

        memory.clear_memory()

        print("=" * 100, flush=True)

        print(f"episode: {i_episode}", flush=True)

        print(f"loss: {-running_reward.mean() / num_steps}", flush=True)

        rew_means.append( # this is what is dumped to the json file
            {
                "rew": (running_reward.mean() / num_steps).item(),
                "opp_rew": (running_opp_reward.mean() / num_steps).item(),
                "M_means": M_means,
            }
        )

        print(f"opponent loss: {-running_opp_reward.mean() / num_steps}", flush=True)

        if i_episode % save_freq == 0:
            ppo.save(os.path.join("runs", name, f"{i_episode}.pth"))
            with open(os.path.join("runs", name, f"out_{i_episode}.json"), "w") as f:
                json.dump(rew_means, f)
            print(f"SAVING! {i_episode}")
        
        # print(prof)

    ppo.save(os.path.join("runs", name, f"{i_episode}.pth"))
    with open(os.path.join("runs", name, f"out_{i_episode}.json"), "w") as f:
        json.dump(rew_means, f)
    with open(os.path.join("runs", name, f"quartile_dumps.json"), "w") as f:
        json.dump(quartile_dumps, f)

    print(f"SAVING! {i_episode}")

    print("Running plot_bigtime.py with filename: ", name)
    
    plot_all(name, caller="ppo", opponent=args.opponent, game=args.game)