import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_rew_vs_ep(p1, p2, values_1, values_2, filename, game, ax=None): # Fixed (for non-mfos).
    '''
    Plots the average reward per episode vs. training episode for two agents, and saves the plot to a file.
    For self-play with annealing, also plots the average reward per episode vs. training episode for the agents vs. a naive-learning agent.

    May look jagged for low batch size. 
    '''

    figname = f"{p1}vs{p2}_{game}_rew_vs_ep.png"
    
    if p1 == p2:
        p1 = f"{p1} 0"
        p2 = f"{p2} 1"
    
    later_ax = 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        later_ax = None

    ax.plot(values_1, color='blue', label=f'{p1} Reward')
    ax.plot(values_2, color='red', label=f'{p2} Reward')
    
    # Add labels and legend
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Reward')
    ax.legend()

    filename = filename.replace("/", "_")

    ax.set_title(f'{p1} vs. {p2}: {game} Rewards vs. Training Episode')

    if not os.path.isdir(f"images/{filename}"):
        os.mkdir(f"images/{filename}")
    # Show the plot
    if later_ax is None:
        fig.savefig(f'images/{filename}/{figname}')
        plt.clf()
        plt.close()


def main(filename):
    with open(f'runs/{filename}/out.json', 'r') as file:
        results = json.load(file)

    # Extracting unique values for games, p1s, and p2s
    games = list(set(result['game'] for result in results))
    p1s = list(set(result['p1'] for result in results))
    p2s = list(set(result['p2'] for result in results))

    # Iterating over games, p1s, and p2s to create subplots
    import numpy as np

    for game in games:
        fig, axs = plt.subplots(len(p1s), len(p2s), figsize=(10, 10))
        for i, p1 in enumerate(p1s):
            for j, p2 in enumerate(p2s):
                ax = axs[i][j]
                ax.set_title(f"p1={p1}, p2={p2}")  # Setting the title of the subplot
                eps_values = []
                all_rewards_1 = []
                all_rewards_2 = []
                plotted_labels = []

                # Extracting rew_0, rew_1, and eps values for the current combination
                for result in results:
                    if result['game'] == game and result['p1'] == p1 and result['p2'] == p2:
                        all_rewards_1.append(result['rewards_1'])
                        all_rewards_2.append(result['rewards_2'])
                        eps_values.append(result['eps'])

                # Plotting rew_0 and rew_1 for each eps value
                for k, eps in enumerate(eps_values):
                    if eps <= 2:  # do not plot for eps > 2
                        alpha_val = 1 if eps == 0 else 0.3 + 1/(80*eps+2)  # alpha gradient
                        label_1 = f"p1 w/ +\u03B5={eps}"
                        label_2 = f"p2 w/ +\u03B5={eps}"
                        color_1 = 'red' if eps == 0 else 'pink'
                        color_2 = 'blue' if eps == 0 else 'purple'
                        if label_1 not in plotted_labels:  # avoid duplicate legends
                            ax.plot(all_rewards_1[k], alpha=alpha_val, color=color_1, label=label_1)
                            plotted_labels.append(label_1)
                        else:
                            ax.plot(all_rewards_1[k], alpha=alpha_val, color=color_1)
                        if label_2 not in plotted_labels:  # avoid duplicate legends
                            ax.plot(all_rewards_2[k], alpha=alpha_val, color=color_2, label=label_2)
                            plotted_labels.append(label_2)
                        else:
                            ax.plot(all_rewards_2[k], alpha=alpha_val, color=color_2)

                ax.set_xlabel(f"Training Episode")  # Setting x-axis label
                ax.set_ylabel("Reward")  # Setting y-axis label

        # Make a global legend
        handles, labels = axs[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')


        fig.suptitle(f"{game}: Rewards vs Training episode, varying +\u03B5 incentive for p2 to take act2")  # Setting the title of the figure
        fig.tight_layout()  # Adjusting subplot layout
        dir_name = f'images/{filename}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(f'images/{filename}/{game}_full_asymms.png')  # Saving the plots
        plt.close()

if __name__ == "__main__":
    main("non_mfos_asymms_oneshot_full")