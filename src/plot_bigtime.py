import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import os
import numpy as np

## Pseudo code to fill in later: 
'''
def main(filename):
    this parses the filename to determine which data to load
        if the filename contains "non_mfos", then it is a round robin set, and each entry in data is a separate game
        this means each entry calls the plot functions. 
        if the filename contains "self" and doesnt contain "nolambda" then I need to additionally load ep values, vnl values
        to plot the mfos vs mfos and mfos vs nl properly
        if the filename contains "nolambda" then I can pretty much plot as normal. just need to get the data correctly.

    then loads the corresponding data. 
    
    regardless of the filename, I need/want: p1, p2, game, and then end_params, all_params, all_Ms, rewards_1, rewards_2, or equivalent.

    this data is then fed into a subset of a few plotting functions
    each of which plots a different graph I want, and saves it to a file
        one of p(act) at the end -- see lola 
            plot_p_act
        one of p(act) over the course of the training run -- only of p2

        one of E(reward) over the course of the training run
            plot_rew_vs_ep
        one of E(state visitation) -- only four states. 
            plot_esv

'''


def plot_rew_vs_ep(p1, p2, values_1, values_2, filename, game, ep_values=[], nl_ep_values=[], rew_vnl_values=[], opp_vnl_rew_values=[]): # Fixed (for non-mfos).
    '''
    Plots the average reward per episode vs. training episode for two agents, and saves the plot to a file.
    For self-play with annealing, also plots the average reward per episode vs. training episode for the agents vs. a naive-learning agent.

    May look jagged for low batch size. 
    '''
    
    if filename.find("self") != -1: self = True
    else: self = False
    if filename.find("nolambda") != -1: nolambda = True
    else: nolambda = False

    figname = f"{p1}vs{p2}_{game}_rew_vs_ep.png"
    
    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"

    if not self:
        plt.plot(values_1, color='blue', label=f'{p1} Reward')
        plt.plot(values_2, color='red', label=f'{p2} Reward')
    else: # MFOS vs MFOS
        plt.plot(ep_values, values_1, color='blue', label=f'{p1} Reward')
        plt.plot(ep_values, values_2, color='red', label=f'{p2} Reward')
        if not nolambda:
            plt.plot(nl_ep_values, rew_vnl_values, color='tab:blue', label=f'{p1} vs NL Reward')
            plt.plot(nl_ep_values, opp_vnl_rew_values, color='tab:red', label=f'{p2} vs NL Reward')
    
    # Add labels and legend
    plt.xlabel('Training Episode')
    plt.ylabel('Reward')
    plt.legend()

    filename = filename.replace("/", "_")

    id = random.randint(1000, 9999)

    plt.title(f'{p1} vs. {p2}: {game} Rewards vs. Training Episode')

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")
    # Show the plot
    plt.savefig(f'images/{filename}/{figname}')
    plt.clf()


def plot_p_act(p1, p2, end_params, game, filename): # Fixed (for non-mfos). 
    '''
    Plots the probability/threshold of each agent taking action at the end of training, and saves the plot to a file.
    Only plots the first 50 of the batch.
    '''
    
    if game.find("diff") != -1: diff = True
    else: diff = False

    states = ["s_0"]
    p2_nums = [0]
    if game.find("I") != -1:
        p2_nums.extend([1,3,2,4])
        if game.find("PD") != -1:
            states.extend(["CC","CD","DC","DD"])
        elif game.find("MP") != -1:
            states.extend(["HH","HT","TH","TT"])
        elif game.find("HD") != -1:
            states.extend(["SwSw", "SwSt", "StSw", "StSt"])
        elif game.find("SH") != -1:
            states.extend(["SS", "SH", "HS", "HH"])
    if game.find("PD") != -1: act = "Cooperate"
    elif game.find("MP") != -1: act = "Heads"
    elif game.find("HD") != -1: act = "Swerve"
    elif game.find("SH") != -1: act = "Stag"
    else: print("Invalid game")

    scatter_points = {0: [], 1: [], 2: [], 3: [], 4: []}

    figname = f"{p1}vs{p2}_{game}_p_act.png"

    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"

    for i in range(len(end_params[0])):
        agent1 = end_params[0][i]
        agent2 = end_params[1][i]
        for j, k in enumerate(p2_nums): # TODO Check. 
            probs_1 = agent1[j]
            probs_2 = agent2[k]
            scatter_points[j].append((probs_1, probs_2))

    colors = sns.color_palette()
    #colors[0], colors[-1] = colors[-1], colors[0] # making it match LOLA

    for j, color in zip(range(len(states)), colors):
        x, y = zip(*scatter_points[j])
        plt.scatter(x, y, color=color, label=f'{states[j]}', alpha=0.3)

    if diff:
        plt.xlabel(f'{p1} Thresh({act})')
        plt.ylabel(f'{p2} Thresh({act})')

    else:
        plt.xlabel(f'{p1} Prob({act})')
        plt.ylabel(f'{p2} Prob({act})')

    plt.grid(True)

    # Set square aspect ratio
    plt.gca().set_aspect('equal')

    # Set axis limits
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    plt.legend()
    plt.title(f'{p1} vs. {p2}: {game} Final Action Parameters')
    
    filename = filename.replace("/", "_")

    id = random.randint(1000, 9999)

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")

    plt.savefig(f'images/{filename}/{figname}')
    plt.clf()


def plot_p_act_vs_ep_med(p1, p2, all_params, game, filename): # Fixed (for non-mfos). 
    '''
    Plots the median probability of taking a certain action vs. training episode for two agents, and saves the plot to a file.
    25th and 75th percentile probabilities are also shaded in.
    '''
    states = ["s_0"]
    p2_nums = [0]
    if game.find("I") != -1:
        all_arr = np.array(all_params).reshape(len(all_params), 3, 5)
        p2_nums.extend([1,3,2,4])
        if game.find("PD") != -1:
            states.extend(["CC","CD","DC","DD"])
        elif game.find("MP") != -1:
            states.extend(["HH","HT","TH","TT"])
        elif game.find("HD") != -1:
            states.extend(["SwSw", "SwSt", "StSw", "StSt"])
        elif game.find("SH") != -1:
            states.extend(["SS", "SH", "HS", "HH"])
    else:
        all_arr = np.array(all_params).reshape(len(all_params), 3, 1)
    if game.find("PD") != -1: act = "Cooperate"
    elif game.find("MP") != -1: act = "Heads"
    elif game.find("HD") != -1: act = "Swerve"
    elif game.find("SH") != -1: act = "Stag"
    else: print("Invalid game")

    prob_or_thresh = "Thresh" if (game.find("diff") != -1) else "Prob"
    
    figname = f"{p1}vs{p2}_{game}_p_act_vs_ep_med.png"

    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"
    
    for i, state in enumerate(states):
        # agent_actions = all_arr[:,:,i]
        episodes = np.arange(all_arr.shape[0])

        # Calculate the median action probability per episode
        median_actions = all_arr[:, 0, i]
        lower_actions = all_arr[:, 1, i]
        upper_actions = all_arr[:, 2, i]
        # Calculate the 25th and 75th percentiles
        # lower_quartile = np.percentile(agent_actions, 25, axis=1)
        # upper_quartile = np.percentile(agent_actions, 75, axis=1)

        plt.plot(episodes, median_actions, label=f'{state}')
        plt.fill_between(episodes, lower_actions, upper_actions, alpha=0.2)

    #for i, state in enumerate(states):
        #plt.plot(all_params_1[:,i], label=f'{p1}, {state}')
        #plt.plot(all_arr_1_maybe[:, i], label=f'{state}')

    plt.xlabel('Training Episode')
    plt.ylabel(f'{prob_or_thresh} ({act}) - {p1}')
    plt.legend()
    plt.title(f'{p1} vs. {p2}: {game} {prob_or_thresh} of {act} vs. Training Episode - {p1}')
    
    filename = filename.replace("/", "_")

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")

    plt.savefig(f'images/{filename}/{figname}')
    plt.clf()


def plot_p_act_vs_ep(p1, p2, all_params, game, filename): # Fixed (for non-mfos). 
    '''
    Plots the probability of taking a certain action vs. training episode for two agents, over an example run (first of the batch). 
    '''
    states = ["s_0"]
    p2_nums = [0]
    if game.find("I") != -1:
        all_arr = np.array(all_params).reshape(len(all_params), 1, 5) 
        p2_nums.extend([1,3,2,4])

        if game.find("PD") != -1:
            states.extend(["CC","CD","DC","DD"])
        elif game.find("MP") != -1:
            states.extend(["HH","HT","TH","TT"])
        elif game.find("HD") != -1:
            states.extend(["SwSw", "SwSt", "StSw", "StSt"])
        elif game.find("SH") != -1:
            states.extend(["SS", "SH", "HS", "HH"])
    else:
        all_arr = np.array(all_params).reshape(len(all_params), 1, 1) 
    if game.find("PD") != -1: act = "Cooperate"
    elif game.find("MP") != -1: act = "Heads"
    elif game.find("HD") != -1: act = "Swerve"
    elif game.find("SH") != -1: act = "Stag"
    else: print("Invalid game")

    prob_or_thresh = "Thresh" if (game.find("diff") != -1) else "Prob"

    figname = f"{p1}vs{p2}_{game}_p_act_vs_ep.png"
    
    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"

    for i, state in enumerate(states):
        # agent_actions = all_arr[:,:,i]
        episodes = np.arange(all_arr.shape[0])

        plt.plot(episodes, all_arr[:, 0, i], label=f'{state}')
        # plt.fill_between(episodes, lower_actions, upper_actions, alpha=0.2)

    plt.xlabel('Training Episode')
    plt.ylabel(f'{prob_or_thresh} ({act}) - {p1}')
    plt.legend()
    plt.title(f'{p1} vs. {p2}: {game} {prob_or_thresh} of {act} vs. Training Episode - {p1}')
    
    filename = filename.replace("/", "_")

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")

    plt.savefig(f'images/{filename}/{figname}')
    plt.clf()


def plot_esv(p1, p2, all_Ms, game, filename): # Fixed (for non-mfos). 
    '''
    Plots the expected state visitation vs. training episode over an example run (first of the batch). 
    '''
    if game.find("PD") != -1:
        states = ["CC","CD","DC","DD"]
    elif game.find("MP") != -1:
        states = ["HH","HT","TH","TT"]
    elif game.find("HD") != -1:
        states = ["SwSw", "SwSt", "StSw", "StSt"]
    elif game.find("SH") != -1:
        states = ["SS", "SH", "HS", "HH"]
    if game.find("PD") != -1: act = "Cooperate"
    elif game.find("MP") != -1: act = "Heads"
    elif game.find("HD") != -1: act = "Swerve"
    elif game.find("SH") != -1: act = "Stag"
    else: print("Invalid game")
    
    figname = f"{p1}vs{p2}_{game}_esv.png"

    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"
    
    all_Ms = np.array(all_Ms).squeeze()

    for i, state in enumerate(states):
        plt.plot(all_Ms[:,i], label=f'{state}')

    plt.xlabel('Training Episode')
    plt.ylabel('Expected State Visitation')
    plt.legend()
    plt.title(f'{p1} vs. {p2}: {game} Expected State Visitation vs. Training Episode')
    
    filename = filename.replace("/", "_")

    id = random.randint(1000, 9999)

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")

    plt.savefig(f'images/{filename}/{figname}')
    plt.clf()


def plot_non_mfos(data, filename):
    for entry in data:
        curr_game = entry["game"]
        curr_p1 = entry["p1"]
        curr_p2 = entry["p2"]
        curr_end_params = entry["end_params"]
        curr_all_params = entry["all_params_1"]
        curr_rew_1 = entry["rewards_1"]
        curr_rew_2 = entry["rewards_2"]

        plt.clf()
        plot_rew_vs_ep(p1=curr_p1, p2=curr_p2, game=curr_game, values_1=curr_rew_1, values_2=curr_rew_2, filename=filename)
        plot_p_act(p1=curr_p1, p2=curr_p2, end_params=curr_end_params, game=curr_game, filename=filename) # need to fix the colors/legend. 
        plot_p_act_vs_ep(p1=curr_p1, p2=curr_p2, all_params=curr_all_params, game=curr_game, filename=filename)
        
        if curr_game.find("I") != -1: # one-shot games do not have all_Ms/expected state visitation
            curr_Ms = entry["all_Ms"]
            plot_esv(p1=curr_p1, p2=curr_p2, all_Ms=curr_Ms, game=curr_game, filename=filename)


def plot_self(data, filename):
    for entry in data:
        curr_game = entry["game"]



def main(filename, caller=None):
    for ext in ["_8192", "_4096", "_2048", "_1024", "_512", "_256", "_128", "_64", "_32", "_16", "_8", ""]:
        if os.path.isfile(f"runs/{filename}/out{ext}.json"):
            filename = f"{filename}/out{ext}"
            break

    with open(f'runs/{filename}.json', 'r') as file:
        data = json.load(file)

    if caller=="non_mfos":
        plot_non_mfos(data, filename)
        # end the program here
        quit()
    elif caller=="self":
        plot_self(data, filename)
        # end the program here
        quit()

    if filename.find("non_mfos") != -1:
        plot_non_mfos(data, filename)
        # end the program here
        quit()

    if filename.find("self") != -1:
        plot_self(data, filename)

    quit()


    if filename.find("diff_ipd") != -1:
        game = "diffIPD"
    elif filename.find("diff_chk") != -1:
        game = "diffCHK"
    elif filename.find("diff_imp") != -1:
        game = "diffIMP"
    elif filename.find("chicken") != -1:
        game = "chicken"
    elif filename.find("imp") != -1:
        game = "IMP"


    if filename.find("self") != -1:
        # this means that the filename contains "self"
        self = True
    if filename.find("nl") != -1:
        # this means that the filename contains "self"
        opponent = "NL "
    if filename.find("nolambda") != -1:
        nolambda = True
    else:
        nolambda = False
    
    # plot_nons(end_params, game, num_episodes=50, diff=False)

# parser = argparse.ArgumentParser()
# parser.add_argument('--filename', type=str, required=False)
# args = parser.parse_args()

if __name__ == "__main__":
    filename = "non_mfos_all_diffoneshot_test1"
    main(filename)
    quit()

    if filename.find("self") != -1:
        # this means that the filename contains "self"
        self = True

    if filename.find("nl") != -1:
        # this means that the filename contains "self"
        opponent = "NL "

    if filename.find("nolambda") != -1:
        nolambda = True
    else:
        nolambda = False

    # Read the JSON file
    with open(f'runs/{filename}.json', 'r') as file:
        data = json.load(file)

    # Extract the values
    ep_values = []
    rew_values = []
    rew_vnl_values = []
    nl_ep_values = []
    opp_rew_values = []
    opp_vnl_rew_values = []

    if self:
        for entry in data:
            if(entry["other"]):
                ep_values.append(entry["ep"])
                rew_values.append(entry["rew 0"])
                opp_rew_values.append(entry["rew 1"])
            else:
                nl_ep_values.append(entry["ep"])
                rew_vnl_values.append(entry["rew 0"])
                opp_vnl_rew_values.append(entry["rew 1"])
    else:
        for entry in data:
            rew_values.append(entry["rew"])
            opp_rew_values.append(entry["opp_rew"])

    

    # Plot the data
    if not self:
        plt.plot(rew_values, color='blue', label='M-FOS Reward')
        plt.plot(opp_rew_values, color='red', label=f'Opponent {opponent}Reward')
    else:
        plt.plot(ep_values, rew_values, color='blue', label='M-FOS 0 Reward')
        plt.plot(ep_values, opp_rew_values, color='red', label='M-FOS 1 Reward')
        if not nolambda:
            plt.plot(nl_ep_values, rew_vnl_values, color='tab:blue', label='M-FOS 0 vs NL Reward')
            plt.plot(nl_ep_values, opp_vnl_rew_values, color='tab:red', label='M-FOS 1 vs NL Reward')

    # Add labels and legend
    plt.xlabel('Training Episode')
    plt.ylabel('Reward')
    plt.legend()

    # change figure name from Figure 1 to Figure 2
    filename = filename.replace("/", "_")

    plt.title(f'{filename}')

    # Show the plot
    plt.savefig(f'images/{filename}.png')
    plt.show()



# ignore


def plot_nons(end_params, env, num_episodes=50, diff=False):
    if env=="IPD" or env =="diffIPD":
        states=["CC","CD","DC","DD","s_0"]
        act = "Cooperate"
    elif env=="CHK" or env=="IHD":
        states = ["SwSw","SwSt","StSw","StSt","s_0"]
        act = "Swerve"
    elif env=="IMP":
        states = ["HH","HT","TH","TT","s_0"]
        act = "Heads"
    else: print("Invalid game")

    scatter_points = {0: [], 1: [], 2: [], 3: [], 4: []}

    for i in range(0, num_episodes, 2):
        agent1 = end_params[i]
        agent2 = end_params[i+1]
        for j in range(env.NUM_STATES):
            logits_1 = agent1[j]
            probs_1 = torch.sigmoid(logits_1)
            logits_2 = agent2[j]
            probs_2 = torch.sigmoid(logits_2)

            scatter_points[j].append((probs_1.item(), probs_2.item()))

    colors = sns.color_palette()
    #colors[0], colors[-1] = colors[-1], colors[0] # making it match LOLA

    for j, color in zip(range(env.NUM_STATES), colors):
        x, y = zip(*scatter_points[j])
        plt.scatter(x, y, color=color, label=f'{states[j]}', alpha=0.3)

    if diff:
        plt.xlabel(f'Agent 1 Thresh({act})')
        plt.ylabel(f'Agent 2 Thresh({act})')
        plt.title('Agent Thresholds')
    else:
        plt.xlabel(f'Agent 1 Prob({act})')
        plt.ylabel(f'Agent 2 Prob({act})')
        plt.title('Agent Probabilities')
    plt.grid(True)
    plt.legend()

    # Set square aspect ratio
    plt.gca().set_aspect('equal')

    # Set axis limits
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.show()

