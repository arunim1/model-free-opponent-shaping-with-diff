import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cycler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="non_mfos_pd_widelrs_diffabs_G1.75")
args = parser.parse_args()

# Load the JSON file
folder = args.folder
with open(f"./{folder}/out.json", "r") as f:
    data = json.load(f)

# Extract unique player combinations
player_combos = list(set((d["p1"], d["p2"]) for d in data))

# We'll average over the last 20% of each run
pct_to_average = 0.2

# plt.rcParams["axes.prop_cycle"] = cycler(color=colors)


# Create a plot for each player combination
for combo in player_combos:
    p1, p2 = combo

    # Filter data for the current player combination
    combo_data = [d for d in data if d["p1"] == p1 and d["p2"] == p2]

    # Extract learning rates and average ESVs
    lrs = [d["lr"] for d in combo_data]
    avg_ms = [np.array(d["avg_Ms"]) for d in combo_data]
    ms = [np.array(d["Ms"]) for d in combo_data]

    num_stepss = [d["num_steps"] for d in combo_data]

    avg_esvs = [
        np.mean(m[int(num_steps * (1 - pct_to_average)) :], axis=0)
        for m, num_steps in zip(avg_ms, num_stepss)
    ]
    esvs = [
        np.mean(m[int(num_steps * (1 - pct_to_average)) :], axis=0)
        for m, num_steps in zip(ms, num_stepss)
    ]

    # Each avg_esv is of shape (4,) -- corresponding to [CC, DC, CD, DD] for the PD
    # Each esv is of shape (20, 4) -- corresponding to [CC, DC, CD, DD] for the PD for each run tracked

    # Sort the data based on learning rates
    sorted_data = sorted(zip(lrs, avg_esvs))
    lrs, avg_esvs = zip(*sorted_data)
    sorted_data2 = sorted(zip(lrs, esvs))
    lrs, esvs = zip(*sorted_data2)

    # Plot the data
    plt.figure(figsize=(8, 8))
    states = ["CC", "DC", "CD", "DD"]
    colors = list(plt.cm.tab10.colors)
    colors[1], colors[2] = colors[2], colors[1]  # Swap the 2nd and 3rd colors

    for i in range(4):
        state_avg_esvs = [avg_esv[i] for avg_esv in avg_esvs]
        plt.plot(lrs, state_avg_esvs, label=states[i], alpha=0.8, color=colors[i])

    plt.legend(loc="upper right")

    for i in range(4):
        state_esvs = [esv[i] for esv in esvs]
        plt.plot(lrs, state_esvs, label=states[i], alpha=0.13, color=colors[i])

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Average Expected State Visitation")
    plt.title(f"Average ESV vs. Learning Rate ({p1} vs. {p2})")
    plt.grid(False)

    # Create payoff matrix data
    payoff_mat_p1 = combo_data[0]["payoff_mat_p1"]
    assert all(
        combo_data[i]["payoff_mat_p1"] == payoff_mat_p1
        for i in range(1, len(combo_data))
    )
    payoff_mat_p2 = combo_data[0]["payoff_mat_p2"]
    assert all(
        combo_data[i]["payoff_mat_p2"] == payoff_mat_p2
        for i in range(1, len(combo_data))
    )
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

    # Add payoff matrices to the plot
    table = plt.table(
        cellText=payoff_data,
        cellLoc="center",
        loc="upper left",
        colWidths=[0.05, 0.12, 0.12],
    )
    table.scale(1, 1.7)  # Adjust the size of the payoff matrices
    table.set_fontsize(13)  # Adjust the font size of the payoff matrices

    plt.tight_layout()
    if not os.path.exists(f"./{folder}/img"):
        os.mkdir(f"./{folder}/img")
    plt.savefig(f"./{folder}/img/avg_esv_vs_lr_{p1}_{p2}.png")
    plt.close()
