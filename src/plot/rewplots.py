import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder", type=str, default="runs/non_mfos_pd_midlrs_diffsquared_G1.75"
)
args = parser.parse_args()

# Load the JSON file
folder = args.folder
with open(f"./{folder}/out.json", "r") as f:
    data = json.load(f)

# Extract unique player combinations
player_combos = list(set((d["p1"], d["p2"], d["lr"]) for d in data))

timesteps_to_plot = 5

# Create a plot for each player combination
for combo in player_combos:
    p1, p2, lr = combo

    # Filter data for the current player combination
    combo_data = [d for d in data if d["p1"] == p1 and d["p2"] == p2 and d["lr"] == lr]

    assert len(combo_data) == 1
    d = combo_data[0]

    avg_rewards_1 = d["avg_rewards_p1"]
    avg_rewards_2 = d["avg_rewards_p2"]

    avg_rewards_1 = np.array(avg_rewards_1)
    avg_rewards_2 = np.array(avg_rewards_2)

    rewards_1 = d["rewards_p1"]
    rewards_2 = d["rewards_p2"]

    rewards_1 = np.array(rewards_1)
    rewards_2 = np.array(rewards_2)

    num_steps = d["num_steps"]

    # print all the shapes
    # print(avg_rewards_1.shape)
    # print(avg_rewards_2.shape)
    # print(rewards_1.shape)
    # print(rewards_2.shape)

    # (500,)
    # (500,)
    # (500, 20)
    # (500, 20)

    colors = ["tab:blue", "tab:red"]

    # make it a square
    plt.figure(figsize=(8, 8))

    x_vals = range(num_steps)
    plt.plot(x_vals, avg_rewards_1, label=f"{p1}", alpha=0.8, color=colors[0])
    plt.plot(x_vals, avg_rewards_2, label=f"{p2}", alpha=0.8, color=colors[1])
    plt.legend()

    plt.plot(x_vals, rewards_1, label=f"{p1}", alpha=0.05, color=colors[0])
    plt.plot(x_vals, rewards_2, label=f"{p2}", alpha=0.05, color=colors[1])

    plt.xlabel(f"Timestep")
    plt.ylabel(f"Reward")
    plt.title(f"Average Reward vs. Timestep")

    plt.grid(False)

    # Create payoff matrix data
    payoff_mat_p1 = d["payoff_mat_p1"]
    payoff_mat_p2 = d["payoff_mat_p2"]

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
        colWidths=[0.1, 0.24, 0.24],
    )
    table.scale(1, 1.7)  # Adjust the size of the payoff matrices
    table.set_fontsize(13)  # Adjust the font size of the payoff matrices

    plt.tight_layout()
    if not os.path.exists(f"./{folder}/img"):
        os.mkdir(f"./{folder}/img")
    plt.savefig(f"./{folder}/img/reward_{p1}_{p2}_{lr}.png")
    plt.close()
