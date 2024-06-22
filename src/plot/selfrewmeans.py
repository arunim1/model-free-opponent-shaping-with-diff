import json
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--folder", type=str, default="/runs/G2multiselfbs1024lowlrnoanneal"
)
args = parser.parse_args()

# Load the JSON file
folder = args.folder
with open(f"./{folder}/out.json", "r") as f:
    data = json.load(f)

# Extract unique player combinations
player_combos = list(set((d["p1"], d["p2"]) for d in data))

d = data[0]

rewmeans = d["rew_means"]

del d

x_v_nl = [rew["ep"] for rew in rewmeans if not rew["other"]]
y_v_nl = [rew["rew 0"] for rew in rewmeans if not rew["other"]]
plt.plot(x_v_nl, y_v_nl, label="p1 vs NL", alpha=0.4, color="tab:blue")
y_v_nl = [rew["rew 1"] for rew in rewmeans if not rew["other"]]
plt.plot(x_v_nl, y_v_nl, label="p2 vs NL", alpha=0.4, color="tab:red")

x_v_non_nl = [rew["ep"] for rew in rewmeans if rew["other"]]
y_v_non_nl = [rew["rew 0"] for rew in rewmeans if rew["other"]]
plt.plot(x_v_non_nl, y_v_non_nl, label="p1", alpha=0.8, color="tab:blue")
y_v_non_nl = [rew["rew 1"] for rew in rewmeans if rew["other"]]
plt.plot(x_v_non_nl, y_v_non_nl, label="p2", alpha=0.8, color="tab:red")

plt.legend()
plt.xlabel("Training Episode")
plt.ylabel("Reward")
plt.title("Average Reward vs. Training Episode")
plt.grid(False)

if not os.path.exists(f"./{folder}/img"):
    os.mkdir(f"./{folder}/img")
plt.savefig(f".{folder}/img/avg_rew_vs_ep.png")
plt.close()
