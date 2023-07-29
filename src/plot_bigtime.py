import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import numpy as np
import argparse


def plot_rew_vs_ep(p1, p2, values_1, values_2, filename, game, ax=None, timestep=None, ep_values=[], nl_ep_values=[], rew_vnl_values=[], opp_vnl_rew_values=[]): 
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
    
    later_ax = 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        later_ax = None

    if not self:
        ax.plot(values_1, color='blue', label=f'{p1} Reward')
        ax.plot(values_2, color='red', label=f'{p2} Reward')
    else: # MFOS vs MFOS
        ax.plot(ep_values, values_1, color='blue', label=f'{p1} Reward')
        ax.plot(ep_values, values_2, color='red', label=f'{p2} Reward')
        if not nolambda:
            ax.plot(nl_ep_values, rew_vnl_values, color='tab:blue', label=f'{p1} vs NL Reward')
            ax.plot(nl_ep_values, opp_vnl_rew_values, color='tab:red', label=f'{p2} vs NL Reward')
    
    # Add labels and legend
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Reward')
    ax.legend()

    filename = filename.replace("/", "_")

    ax.set_title(f'{p1} vs. {p2}: {game} Rewards vs. Training Episode')
    if timestep is not None:
        ax.set_title(f'Timestep: {timestep}')
    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")
    # Show the plot
    if later_ax is None:
        fig.savefig(f'images/{filename}/{figname}')
        plt.clf()
        plt.close()


def plot_p_act(p1, p2, end_params, game, filename, ax=None, timestep=None):  
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
    else: act = "??", print("Invalid game")

    scatter_points = {0: [], 1: [], 2: [], 3: [], 4: []}

    figname = f"{p1}vs{p2}_{game}_p_act.png"

    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"

    later_ax = 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        later_ax = None

    for i in range(len(end_params[0])):
        agent1 = end_params[0][i]
        agent2 = end_params[1][i]
        for j, k in enumerate(p2_nums): # TODO Check. 
            probs_1 = agent1[j]
            probs_2 = agent2[k]
            scatter_points[j].append((probs_1, probs_2))

    colors = sns.color_palette()

    for j, color in zip(range(len(states)), colors):
        x, y = zip(*scatter_points[j])
        ax.scatter(x, y, color=color, label=f'{states[j]}', alpha=0.3)

    if diff:
        ax.set_xlabel(f'{p1} Thresh({act})')
        ax.set_ylabel(f'{p2} Thresh({act})')

    else:
        ax.set_xlabel(f'{p1} Prob({act})')
        ax.set_ylabel(f'{p2} Prob({act})')

    ax.grid(True)

    # Set square aspect ratio
    ax.set_aspect('equal', 'box')

    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    ax.legend()
    ax.set_title(f'{p1} vs. {p2}: {game} Final Action Parameters')
    if timestep is not None:
        ax.set_title(f'Timestep: {timestep}')
    filename = filename.replace("/", "_")

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")
    if later_ax is None:
        fig.savefig(f'images/{filename}/{figname}')
        plt.clf()
        plt.close()


def plot_p_act_vs_ep(p1, p2, all_params, game, filename, ax=None, timestep=None):  
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
    else: act = "??", print("Invalid game")

    prob_or_thresh = "Thresh" if (game.find("diff") != -1) else "Prob"

    figname = f"{p1}vs{p2}_{game}_p_act_vs_ep.png"
    
    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"
    later_ax = 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        later_ax = None

    for i, state in enumerate(states):
        # agent_actions = all_arr[:,:,i]
        episodes = np.arange(all_arr.shape[0])

        ax.plot(episodes, all_arr[:, 0, i], label=f'{state}')
        # plt.fill_between(episodes, lower_actions, upper_actions, alpha=0.2)

    ax.set_xlabel('Training Episode')
    if timestep is None: ax.set_ylabel(f'{prob_or_thresh} ({act}) - {p1}') # proxy for MFOS/Meta game 
    else: ax.set_ylabel(f'{prob_or_thresh} ({act}) - {p2}') 
    
    offset = 1 * 0.05  # 5% offset
    ax.set_ylim(-offset, 1 + offset)
    ax.legend()
    ax.set_title(f'{p1} vs. {p2}: {game} {prob_or_thresh} of {act} vs. Training Episode - {p1}')
    if timestep is not None:
        ax.set_title(f'Timestep: {timestep}')
    filename = filename.replace("/", "_")

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")
    if later_ax is None:
        fig.savefig(f'images/{filename}/{figname}')
        plt.clf()
        plt.close()


def plot_esv(p1, p2, all_Ms, game, filename, ax=None, timestep=None):  
    '''
    Plots the expected state visitation vs. training episode over an example run (first of the batch). 
    '''
    states = []
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
    
    later_ax = 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        later_ax = None
    all_Ms = np.array(all_Ms).squeeze()

    # Calculate sum across each row
    row_sums = all_Ms.sum(axis=1)

    for i, state in enumerate(states):
        ax.plot(all_Ms[:,i], label=f'{state}')

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Expected State Visitation')

    # Get maximum row sum
    max_row_sum = np.max(row_sums)

    # Set y-axis limits, add small offset to ensure visibility of plotted lines
    offset = max_row_sum * 0.05  # 5% offset
    ax.set_ylim(-offset, max_row_sum + offset)
    ax.legend()

    ax.set_title(f'{p1} vs. {p2}: {game} Expected State Visitation vs. Training Episode')
    if timestep is not None:
        ax.set_title(f'Timestep: {timestep}')
    filename = filename.replace("/", "_")

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")

    if later_ax is None:
        fig.savefig(f'images/{filename}/{figname}')
        plt.clf()
        plt.close()


def plot_non_mfos(data, filename): # FIXED!
    for entry in data:
        curr_game = entry["game"]
        curr_p1 = entry["p1"]
        curr_p2 = entry["p2"]
        curr_rew_1 = entry["rewards_1"]
        curr_rew_2 = entry["rewards_2"]
        curr_Ms = entry["all_Ms"]
        
        if entry["ccdr"]:
            fname = f'{filename}_ccdr'
        else:
            fname = f'{filename}'
        if entry["nn_game"]:
            fname = f'{fname}_nn'
        else:
            fname = f'{fname}'

        plt.clf()
        plot_rew_vs_ep(p1=curr_p1, p2=curr_p2, game=curr_game, values_1=curr_rew_1, values_2=curr_rew_2, filename=fname)
        plot_esv(p1=curr_p1, p2=curr_p2, all_Ms=curr_Ms, game=curr_game, filename=fname)

        if entry["end_params"] is not None and entry["all_params_1"] is not None:
            curr_end_params = entry["end_params"]
            curr_all_params = entry["all_params_1"]
            plot_p_act(p1=curr_p1, p2=curr_p2, end_params=curr_end_params, game=curr_game, filename=fname)
            plot_p_act_vs_ep(p1=curr_p1, p2=curr_p2, all_params=curr_all_params, game=curr_game, filename=fname)
        

def plot_self(data, quarts, filename, game, nn_game=True):
    p1 = "MFOS 0"
    p2 = "MFOS 1"
    prob_or_thresh = "Thresh" if game.find("diff") != -1 else "Prob"
    act = "Cooperate" if game.find("PD") != -1 else "Heads" if game.find("MP") != -1 else "Swerve" if game.find("HD") != -1 else "Stag"
    # plotting the reward, opp_rew vs. time
    ep_values = []
    nl_ep_values = []
    rew1 = []
    rew2 = []
    rew_vnl_values = []
    opp_vnl_rew_values = []
    for entry in data:
        if(entry["other"]):
            ep_values.append(entry["ep"])
            rew2.append(entry["rew 0"])
            rew1.append(entry["rew 1"])
        else:
            nl_ep_values.append(entry["ep"])
            rew_vnl_values.append(entry["rew 0"])
            opp_vnl_rew_values.append(entry["rew 1"])

    plot_rew_vs_ep(p1, p2, rew1, rew2, filename, game, ep_values=ep_values, nl_ep_values=nl_ep_values, rew_vnl_values=rew_vnl_values, opp_vnl_rew_values=opp_vnl_rew_values)
    filename = filename.replace("/", "_")

    # plotting the rest    
    fig1, ax1 = plt.subplots(1, 5, figsize=(36, 8))
    fig2, ax2 = plt.subplots(1, 5, figsize=(36, 8))
    fig3, ax3 = plt.subplots(1, 5, figsize=(36, 8))
    fig4, ax4 = plt.subplots(1, 5, figsize=(36, 8))

    for i, entry in enumerate(quarts):
        curr_rew_1 = entry["rewards_1"]
        curr_rew_2 = entry["rewards_2"]
        timestep = entry["timestep"]
        if not nn_game:
            curr_end_params = entry["end_params"]
            curr_all_params = entry["all_params_1"]
            plot_p_act(p1, p2, curr_end_params, game, filename, ax1[i], timestep = timestep)
            plot_p_act_vs_ep(p1, p2, curr_all_params, game, filename, ax2[i], timestep = timestep)
        plot_rew_vs_ep(p1, p2, curr_rew_1, curr_rew_2, game, filename, ax3[i], timestep = timestep) 
        plot_esv(p1, p2, entry["all_Ms"], game, filename, ax4[i], timestep = timestep)
    
    fig1.suptitle(f'{p1} vs. {p2}: {game} Final Params vs. Timestep')
    fig2.suptitle(f'{p1} vs. {p2}: {game} {prob_or_thresh} of {act} vs. Timestep - {p2}')
    fig3.suptitle(f'{p1} vs. {p2}: {game} Reward vs. Timestep')
    fig4.suptitle(f'{p1} vs. {p2}: {game} Expected State Visitation vs. Timestep')
    for fig in [fig1, fig2, fig3, fig4]: fig.tight_layout(pad=2)
    if not nn_game:
        fig1.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_p_act.png')
        fig2.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_p_act_vs_ep.png')
    fig3.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_rew_vs_ep.png')
    fig4.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_esv.png') 


def plot_ppo(data, quarts, filename, p1, p2, game, nn_game=True):
    prob_or_thresh = "Thresh" if game.find("diff") != -1 else "Prob"
    act = "Cooperate" if game.find("PD") != -1 else "Heads" if game.find("MP") != -1 else "Swerve" if game.find("HD") != -1 else "Stag"
    # plotting the reward, opp_rew vs. time
    rew1 = []
    rew2 = []
    for entry in data:
        rew1.append(entry["rew"])
        rew2.append(entry["opp_rew"])
    plot_rew_vs_ep(p1, p2, rew1, rew2, filename, game)
    filename = filename.replace("/", "_")

    # plotting the rest    
    fig1, ax1 = plt.subplots(1, 5, figsize=(36, 8))
    fig2, ax2 = plt.subplots(1, 5, figsize=(36, 8))
    fig3, ax3 = plt.subplots(1, 5, figsize=(36, 8))
    fig4, ax4 = plt.subplots(1, 5, figsize=(36, 8))
    
    for i, entry in enumerate(quarts):
        curr_rew_1 = entry["rewards_1"]
        curr_rew_2 = entry["rewards_2"]
        timestep = entry["timestep"]
        if not nn_game:
            curr_end_params = entry["end_params"]
            curr_all_params = entry["all_params_1"]
            plot_p_act(p1, p2, curr_end_params, game, filename, ax1[i], timestep = timestep)
            plot_p_act_vs_ep(p1, p2, curr_all_params, game, filename, ax2[i], timestep = timestep)
        plot_rew_vs_ep(p1, p2, curr_rew_1, curr_rew_2, game, filename, ax3[i], timestep = timestep) 
        plot_esv(p1, p2, entry["all_Ms"], game, filename, ax4[i], timestep = timestep)
    
    fig1.suptitle(f'{p1} vs. {p2}: {game} Final Params vs. Timestep')
    fig2.suptitle(f'{p1} vs. {p2}: {game} {prob_or_thresh} of {act} vs. Timestep - {p2}')
    fig3.suptitle(f'{p1} vs. {p2}: {game} Reward vs. Timestep')
    fig4.suptitle(f'{p1} vs. {p2}: {game} Expected State Visitation vs. Timestep')
    for fig in [fig1, fig2, fig3, fig4]: fig.tight_layout(pad=2)
    if not nn_game:
        fig1.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_p_act.png')
        fig2.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_p_act_vs_ep.png')
    fig3.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_rew_vs_ep.png')
    fig4.savefig(f'images/{filename}/qs_{p1}vs{p2}_{game}_esv.png') 


def main(filename, caller=None, opponent=None, game=None, nn_game=True):
    filename_out = None
    for ext in ["_8192", "_4096", "_2048", "_1024", "_512", "_256", "_128", "_64", "_32", "_16", "_8", ""]:
        if os.path.isfile(f"runs/{filename}/out{ext}.json"):
            filename_out = f"{filename}/out{ext}"
            break
    if filename_out is None:
        print("No file found")
        quit()

    with open(f'runs/{filename_out}.json', 'r') as file:
        data = json.load(file)

    if caller=="non_mfos":
        plot_non_mfos(data, filename_out)
        quit()
    elif caller=="ppo":
        with open(f'runs/{filename}/quartile_dumps.json', 'r') as file:
            quarts = json.load(file)
        plot_ppo(data, quarts, filename_out, p1 = "MFOS", p2 = opponent, game = game, nn_game=nn_game)
        quit()
    elif caller=="self":
        with open(f'runs/{filename}/quartile_dumps.json', 'r') as file:
            quarts = json.load(file)
        plot_self(data, quarts, filename_out, game=game, nn_game=nn_game)
        quit()

    print("Invalid caller")
    quit()

# arge = argparse.ArgumentParser()
# arge.add_argument("--filename", type=str, default="self_IPD_test2")
# arge.add_argument("--game", type=str, default="IPD")
# arge.add_argument("--opponent", type=str, default="NL")
# arge.add_argument("--caller", type=str, default=None)
# args = arge.parse_args()

if __name__ == "__main__":
    # filename = args.filename
    # game = args.game
    # opponent = args.opponent
    # caller = args.caller
    # main(filename, caller, opponent, game)

    quit()