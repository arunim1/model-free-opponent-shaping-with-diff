import json
import os
import matplotlib.pyplot as plt

def main(filename):
    with open(f'runs/{filename}/out.json', 'r') as file:
        results = json.load(file)
        
    # Extracting unique values for games, p1s, and p2s
    games = list(set(result['game'] for result in results))
    p1s = list(set(result['p1'] for result in results))
    p2s = list(set(result['p2'] for result in results))

    # Creating subplots based on the number of games
    num_games = len(games)
    num_subplots = len(p1s) * len(p2s)
    num_rows = num_games
    num_cols = num_subplots

    # Creating subplots for each game
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    # If there's only one row of games, convert axes to a 2D list for compatibility with the loop
    if num_rows == 1:
        axes = [axes]

    # Iterating over games, p1s, and p2s to create subplots
    for game in games:
        fig, axs = plt.subplots(len(p1s), len(p2s), figsize=(8, 8))
        for i, p1 in enumerate(p1s):
            for j, p2 in enumerate(p2s):
                ax = axs[i][j]
                ax.set_title(f"p1={p1}, p2={p2}")  # Setting the title of the subplot
                rew_0_values = []
                rew_1_values = []
                eps_values = []

                # Extracting rew_0, rew_1, and eps values for the current combination
                for result in results:
                    if result['game'] == game and result['p1'] == p1 and result['p2'] == p2:
                        rew_0_values.append(result['rew_0'])
                        rew_1_values.append(result['rew_1'])
                        eps_values.append(result['eps'])

                # Plotting rew_0 and rew_1 for each eps value
                ax.scatter(eps_values, rew_0_values, label=f"rew {p1}")
                ax.scatter(eps_values, rew_1_values, label=f"rew {p2}")
                ax.set_xscale('symlog', linthresh=0.001)  # Setting x-axis to log scale
                ax.set_xlabel(f"+\u03B5 incentive for p2={p2}")  # Setting x-axis label
                ax.set_ylabel("Final average reward")  # Setting y-axis label
                ax.legend()  # Adding legend to the subplot
        fig.suptitle(f"{game}: Final average rewards vs +\u03B5 incentive for p2 to take act2")  # Setting the title of the figure
        fig.tight_layout()  # Adjusting subplot layout
        dir_name = f'images/{filename}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(f'images/{filename}/{game}_tuning.png')  # Saving the plots
        plt.close()

if __name__ == "__main__":
    main("non_mfos_asymms_iterated_full")
    main("non_mfos_asymms_oneshot_full")