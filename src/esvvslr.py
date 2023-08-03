import matplotlib.pyplot as plt
import numpy as np
import os
import json


def plot_esv(p1, p2, all_Ms, game, filename, lrs, ax=None, timestep=None, M_mean=None):  
    '''
    Plots the expected state visitation vs. training episode over an example run (first of the batch). 
    '''
    states = []
    if game.find("PD") != -1 or game.find("SL") != -1:
        states = ["CC","CD","DC","DD"]
    elif game.find("MP") != -1:
        states = ["HH","HT","TH","TT"]
    elif game.find("HD") != -1:
        states = ["SwSw", "SwSt", "StSw", "StSt"]
    elif game.find("SH") != -1:
        states = ["SS", "SH", "HS", "HH"]
    if game.find("PD") != -1 or game.find("SL") != -1: act = "Cooperate"
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
    lrs = np.array(lrs)

    # Calculate sum across each row
    row_sums = all_Ms.sum(axis=1)

    for i, state in enumerate(states):
        ax.plot(lrs, all_Ms[:,i], label=f'{state}')

    ax.set_xticks(lrs)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Expected State Visitation')

    # Get maximum row sum
    max_row_sum = np.max(row_sums)

    # Set y-axis limits, add small offset to ensure visibility of plotted lines
    offset = max_row_sum * 0.05  # 5% offset
    ax.set_ylim(-offset, max_row_sum + offset)
    ax.legend()

    ax.set_title(f'{p1} vs. {p2}: {game} Average Expected State Visitation vs. Learning Rate')
    if timestep is not None:
        ax.set_title(f'Timestep: {timestep}')
    filename = filename.replace("/", "_")

    # splitting filename by {game} and then setting the first part as the log_filename
    og_filename = filename.split(f"{game}")[0][:-1]

    if not os.path.isdir(f"images/{og_filename}"):
        os.mkdir(f"images/{og_filename}")

    if later_ax is None:
        fig.savefig(f'images/{og_filename}/{filename}')
        plt.clf()
        plt.close()

# plotting mean esv vs. learning rate

def main(filename, caller="non_mfos"):#, game="PD", p1="NL", p2="NL"):
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
        # Initialize an empty dictionary to store (lr, M_mean) tuples for each combination
        all_Ms_dict = {}
        lrs = []

        # Process each item in the data
        for item in data:
            game = item['game']
            p1 = item['p1']
            p2 = item['p2']
            ccdr = item['ccdr']
            nn_game = item['nn_game']
            lr = tuple(item['lr']) if isinstance(item['lr'], list) else item['lr']
            M_mean = np.array(item['M_mean'])

            # Create a unique key for each combination of game, p1, p2, ccdr, and nn_game
            key = (game, p1, p2, ccdr, nn_game)

            # If the key already exists, append the (lr, M_mean) tuple to the existing list, otherwise create a new list
            if key in all_Ms_dict:
                all_Ms_dict[key].append((lr, M_mean))
            else:
                all_Ms_dict[key] = [(lr, M_mean)]

        # Sort the (lr, M_mean) tuples for each combination based on the lr value
        for key in all_Ms_dict.keys():
            if isinstance(all_Ms_dict[key][0][0], list):
                all_Ms_dict[key].sort(key=lambda x: x[0][0])
            else:
                all_Ms_dict[key].sort(key=lambda x: x[0])
            
            lrs = []

            for i in range(len(all_Ms_dict[key])):
                if isinstance(all_Ms_dict[key][i][0], list):
                    lrs.append(all_Ms_dict[key][i][0][0])
                else:
                    lrs.append(all_Ms_dict[key][i][0])
                all_Ms_dict[key][i] = all_Ms_dict[key][i][1]
        
        for key in all_Ms_dict.keys():
            # Convert the dictionary values (lists of (lr, M_mean) tuples) into a NumPy array
            all_Ms_curr = np.array(list(all_Ms_dict[key]))

            # Print the shape of the resulting array
            filename_in =f"{filename}_{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}"
            plot_esv(f"{key[1]}", f"{key[2]}", all_Ms_curr, f"{key[0]}", filename_in, lrs)

    elif caller=="ppo":
        raise NotImplementedError
    else:
        raise NotImplementedError
    
if __name__ == "__main__":
    main("non_mfos_adam_test_IMP_lr_2", caller="non_mfos")