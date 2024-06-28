import json
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="runs/self/delselff272")
args = parser.parse_args()

# Load the JSON file
folder = args.folder

with open(f"./{folder}/run_args.txt", "r") as f:
    run_args = json.load(f)
payoffs = run_args["payoff_mat_p1"][0]
payoffs.extend(run_args["payoff_mat_p1"][1])
payoffs.extend(run_args["payoff_mat_p2"][0])
payoffs.extend(run_args["payoff_mat_p2"][1])

y_min = min(payoffs)
y_max = max(payoffs)

max_num = 0
for f in os.listdir(folder):
    if f.startswith("out_") and f.endswith(".json"):
        num = int(f.split("_")[1].split(".")[0])
        if num > max_num:
            max_num = num
try:
    with open(f"./{folder}/out_{max_num}.json", "r") as f:
        data = json.load(f)
except:
    with open(f"./{folder}/out.json", "r") as f:
        data = json.load(f)

d = data[0]

new = True if "rew_means" in d else False

rewmeans = d["rew_means"] if new else d

plt.figure(figsize=(8, 8))

x_v_nl = [rew["ep"] for rew in rewmeans if not rew["other"]]
y_v_nl = [rew["rew 0"] for rew in rewmeans if not rew["other"]]
plt.plot(
    x_v_nl,
    y_v_nl,
    label="p1 vs NL",
    alpha=0.4,
    color="tab:blue",
    marker="o",
    markersize=2,
)
y_v_nl = [rew["rew 1"] for rew in rewmeans if not rew["other"]]
plt.plot(
    x_v_nl,
    y_v_nl,
    label="p2 vs NL",
    alpha=0.4,
    color="tab:red",
    marker="o",
    markersize=2,
)

x_v_non_nl = [rew["ep"] for rew in rewmeans if rew["other"]]
y_v_non_nl = [rew["rew 0"] for rew in rewmeans if rew["other"]]
plt.plot(
    x_v_non_nl,
    y_v_non_nl,
    label="p1",
    alpha=0.8,
    color="tab:blue",
    marker="o",
    markersize=2,
)
y_v_non_nl = [rew["rew 1"] for rew in rewmeans if rew["other"]]
plt.plot(
    x_v_non_nl,
    y_v_non_nl,
    label="p2",
    alpha=0.8,
    color="tab:red",
    marker="o",
    markersize=2,
)

plt.legend()
plt.xlabel("Training Episode")
plt.ylabel("Reward")
plt.title("Average Reward vs. Training Episode")
plt.grid(False)
plt.ylim(y_min - 0.05, y_max + 0.05)

if not os.path.exists(f"./{folder}/img"):
    os.mkdir(f"./{folder}/img")
plt.savefig(f"./{folder}/img/avg_rew_vs_ep.png")
plt.close()
