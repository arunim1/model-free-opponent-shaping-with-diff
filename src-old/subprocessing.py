import subprocess

base_games = ["PD", "MP", "HD", "SH"]
base_games = ["PD"]
G = None
if G is not None:
    G = str(G)
    G = f"_G{G}"
else:
    G = "testingafterfix"

diff_oneshot = ["diff" + game for game in base_games]
iterated = ["I" + game for game in base_games]
diff_iterated = ["diffI" + game for game in base_games]
all_games = base_games + diff_oneshot + iterated + diff_iterated
opponents = ["NL", "LOLA"] # for MFOS_PPO

# supported modes  are "self_trace", "self_no_tracing", "mfos_ppo"

modes = ["self_trace", "self_no_tracing", "mfos_ppo"]

for game in diff_oneshot + diff_iterated:
    # SELF with no tracing procedure
    if "self_no_tracing" in modes:
        command = [
            "python",
            "src/main_mfos_self.py",
            f"--game={game}",
            f"--exp-name=self_{game}{G}_notrace",
            # "--nn-game",
        ]
        print("Executing command:", " ".join(command))
        result = subprocess.run(command, capture_output=False, text=True)
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print("Command failed with exit code:", result.returncode)
            print("Error output:", result.stderr)
    
    if "self_trace" in modes:
        # SELF with tracing procedure
        command = [
            "python",
            "src/main_mfos_self.py",
            f"--game={game}",
            f"--exp-name=self_{game}{G}_traced",
            f"--tracing",
            # "--nn-game",
        ]
        print("Executing command:", " ".join(command))
        result = subprocess.run(command, capture_output=False, text=True)
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print("Command failed with exit code:", result.returncode)
            print("Error output:", result.stderr)
    
    if "mfos_ppo" in modes:
        # MFOS with PPO
        for opponent in opponents:
            command = [
                "python",
                "src/main_mfos_ppo.py",
                f"--game={game}",
                f"--opponent={opponent}",
                f"--exp-name=mfos_ppo_{game}{G}_{opponent}_nn0.2",
            ]
            print("Executing command:", " ".join(command))
            result = subprocess.run(command, capture_output=False, text=True)
            if result.returncode == 0:
                print("Command executed successfully.")
            else:
                print("Command failed with exit code:", result.returncode)
                print("Error output:", result.stderr)

print("=" * 100, flush=True)
print("Done with all runs")

command = ["gcloud", "compute", "instances", "stop", "instance-arunim", "--zone=us-east1-b"]

print("Executing command:", " ".join(command))
# subprocess.run(command, capture_output=False, text=True)