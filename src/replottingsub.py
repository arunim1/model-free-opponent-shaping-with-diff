import subprocess

base_games = ["PD", "MP", "HD", "SH"]
base_games = ["PD"]

G = 3
if G is not None:
    G = str(G)
    G = f"_G{G}"
else:
    G = ""

Gs = [1.9, 2, 2.1, 2.5, 3]

diff_oneshot = ["diff" + game for game in base_games]
iterated = ["I" + game for game in base_games]
diff_iterated = ["diffI" + game for game in base_games]
all_games = base_games + diff_oneshot + iterated + diff_iterated
opponents = ["NL", "LOLA"] # for MFOS_PPO

# supported modes  are "self_trace", "self_no_tracing", "mfos_ppo"

modes = ["mfos_ppo"]

for G in Gs:
    if G is not None:
        G = str(G)
        G = f"_G{G}"
    for game in all_games:

        if "mfos_ppo" in modes:
            # MFOS with PPO
            for opponent in opponents:
                command = [
                    "python",
                    "src/plot_bigtime.py",
                    f"--filename=mfos_ppo_{game}{G}_{opponent}",
                    f"--game={game}",
                    f"--opponent={opponent}",
                    f"--caller=ppo",
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
