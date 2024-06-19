import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cycler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="/runs/G3multimfospt2")
args = parser.parse_args()

# Load the JSON file
folder = args.folder
with open(f"./{folder}/out.json", "r") as f:
    data = json.load(f)

# Extract unique player combinations
player_combos = list(set((d["p1"], d["p2"]) for d in data))

print(player_combos)

# We'll average over the last 20% of each run
pct_to_average = 0.2

# Create a plot for each player combination
for combo in player_combos:
    p1, p2 = combo

    # Filter data for the current player combination
    combo_data = [d for d in data if d["p1"] == p1 and d["p2"] == p2]

    # Extract learning rates
    lrs = [d["lr"] for d in combo_data]

    fig, axes = plt.subplots(1, 6, figsize=(18, 3), sharey=True)

    for ax in axes:
        ax.set_aspect(1.8)

    for game_idx in range(5):  # Assuming there are always 5 logs in five_game_logs
        avg_ms = [np.array(d["five_game_logs"][game_idx]["avg_Ms"]) for d in combo_data]
        ms = [np.array(d["five_game_logs"][game_idx]["Ms"]) for d in combo_data]
        num_stepss = [d["num_steps"] for d in combo_data]

        avg_esvs = [
            np.mean(m[int(num_steps * (1 - pct_to_average)) :], axis=0)
            for m, num_steps in zip(avg_ms, num_stepss)
        ]
        esvs = [
            np.mean(m[int(num_steps * (1 - pct_to_average)) :], axis=0)
            for m, num_steps in zip(ms, num_stepss)
        ]

        # Sort the data based on learning rates
        sorted_data = sorted(zip(lrs, avg_esvs))
        lrs, avg_esvs = zip(*sorted_data)
        sorted_data2 = sorted(zip(lrs, esvs))
        lrs, esvs = zip(*sorted_data2)

        # Create subplots
        states = ["CC", "DC", "CD", "DD"]
        colors = list(plt.cm.tab10.colors)
        colors[1], colors[2] = colors[2], colors[1]  # Swap the 2nd and 3rd colors

        # state_avg_esvs = [avg_esv[game_idx] for avg_esv in avg_esvs]
        state_avg_esvs = avg_esvs
        state_esvs = esvs

        for i in range(4):
            axes.flatten()[game_idx].plot(
                lrs,
                [state_avg_esv[i] for state_avg_esv in state_avg_esvs],
                label=states[i],
                alpha=0.8,
                color=colors[i],
            )

        axes.flatten()[game_idx].legend(loc="upper right")
        for i in range(4):
            axes.flatten()[game_idx].plot(
                lrs,
                [state_esv[:, i] for state_esv in state_esvs],
                label=states[i],
                alpha=0.08,
                color=colors[i],
            )
        axes.flatten()[game_idx].set_xscale("log")
        axes.flatten()[game_idx].set_xlabel(f"Timestep {game_idx + 1}/5")
        axes.flatten()[game_idx].set_ylim(-0.05, 1.05)

    # set x and y labels for the overall plot
    fig.supxlabel("Learning Rate")
    fig.supylabel("Avg ESV")
    fig.suptitle(f"{p1} vs. {p2}: Average Expected State Visitation vs. Timestep")

    axes.flatten()[-1].axis("off")
    # Add payoff matrices to the plot
    payoff_mat_p1 = combo_data[0]["payoff_mat_p1"]
    payoff_mat_p2 = combo_data[0]["payoff_mat_p2"]
    payoff_data = [
        ["", "C", "D"],
        [
            "C",
            f"{payoff_mat_p1[0][0]}, {payoff_mat_p2[0][0]}",
            f"{payoff_mat_p1[1][0]}, {payoff_mat_p2[1][0]}",
        ],
        [
            "D",
            f"{payoff_mat_p1[0][1]}, {payoff_mat_p2[0][1]}",
            f"{payoff_mat_p1[1][1]}, {payoff_mat_p2[1][1]}",
        ],
    ]
    table = axes.flatten()[-1].table(
        cellText=payoff_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.48, 0.48],
    )
    table.scale(1, 1.7)  # Adjust the size of the payoff matrices
    table.set_fontsize(13)  # Adjust the font size of the payoff matrices

    plt.tight_layout(pad=1.0)
    if not os.path.exists(f"./{folder}/img"):
        os.mkdir(f"./{folder}/img")
    plt.savefig(f"./{folder}/img/avg_esv_vs_lr_{p1}_{p2}_game{game_idx}.png")
    plt.close()
