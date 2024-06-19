import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="non_mfos_pd_highlrs_diffpw32_G3")
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

    avg_params_1 = d["avg_params_p1"]
    avg_params_2 = d["avg_params_p2"]

    avg_params_1 = np.array(avg_params_1)
    avg_params_2 = np.array(avg_params_2)

    params_1 = d["params_p1"]
    params_2 = d["params_p2"]

    params_1 = np.array(params_1)
    params_2 = np.array(params_2)

    num_steps = d["num_steps"]

    pwlinear = d["pwlinear"]
    assert pwlinear is not None

    # get the parameters at timesteps_to_plot number of steps evenly spaced through the trajectory
    timesteps = np.linspace(0, num_steps - 1, timesteps_to_plot)
    timesteps = timesteps.astype(int)

    avg_params_1 = avg_params_1[timesteps]
    avg_params_2 = avg_params_2[timesteps]

    params_1 = params_1[timesteps]
    params_2 = params_2[timesteps]

    # print all the shapes
    # print(avg_params_1.shape)
    # print(avg_params_2.shape)
    # print(params_1.shape)
    # print(params_2.shape)

    # (5, 32)
    # (5, 32)
    # (5, 20, 32)
    # (5, 20, 32)

    # Plot the data in timesteps_to_plot number of subplots (each is square)

    colors = ["tab:blue", "tab:red"]

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        # plot the 32 parameters for each player, as if their x-values were linspace(0, 1, :)
        x_vals = np.linspace(0, 1, pwlinear)
        axs[i].plot(x_vals, avg_params_1[i], label=f"{p1}", alpha=0.8, color=colors[0])
        axs[i].plot(x_vals, avg_params_2[i], label=f"{p2}", alpha=0.8, color=colors[1])
        axs[i].legend()

        axs[i].plot(x_vals, params_1[i].T, label=f"{p1}", alpha=0.05, color=colors[0])
        axs[i].plot(x_vals, params_2[i].T, label=f"{p2}", alpha=0.05, color=colors[1])

        axs[i].set_xlim(-0.05, 1.05)
        axs[i].set_xlabel(f"Diff Value")
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].set_ylabel(f"P(Cooperate)")

    # for i in range(4):
    #     state_avg_esvs = [avg_esv[i] for avg_esv in avg_esvs]
    #     plt.plot(lrs, state_avg_esvs, label=states[i], alpha=0.8, color=colors[i])

    # plt.legend(loc="upper right")

    # for i in range(4):
    #     state_esvs = [esv[i] for esv in esvs]
    #     plt.plot(lrs, state_esvs, label=states[i], alpha=0.13, color=colors[i])

    # plt.title(f"Average P(Cooperate) vs. Diff Value ({p1} vs. {p2})")
    # set title

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
    table = axs[0].table(
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
    plt.savefig(f"./{folder}/img/pwlinear_{p1}_{p2}_{lr}.png")
    plt.close()
