import subprocess

base_games = ["PD", "MP", "HD", "SH"]
base_games = ["PD"]

G = 2.5
if G is not None:
    G = str(G)
    G = f"_G{G}"
else:
    G = ""

diff_oneshot = ["diff" + game for game in base_games]
iterated = ["I" + game for game in base_games]
diff_iterated = ["diffI" + game for game in base_games]
all_games = base_games + diff_oneshot + iterated + diff_iterated
opponents = ["NL", "LOLA"] # for MFOS_PPO

# supported modes  are "self_trace", "self_no_tracing", "mfos_ppo"

modes = ["self_trace", "self_no_tracing", "mfos_ppo"]

for game in all_games:
    # SELF with no tracing procedure
    if "self_no_tracing" in modes:
        command = [
            "python",
            "src/main_mfos_self.py",
            f"--game={game}",
            f"--exp-name=self_{game}{G}_notrace",
        ]
        print("Executing command:", " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
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
            f"--tracing"
        ]
        print("Executing command:", " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
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
                f"--exp-name=mfos_ppo_{game}{G}_{opponent}",
            ]
            print("Executing command:", " ".join(command))
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                print("Command executed successfully.")
            else:
                print("Command failed with exit code:", result.returncode)
                print("Error output:", result.stderr)

print("=" * 100, flush=True)
print("Done with all runs")
