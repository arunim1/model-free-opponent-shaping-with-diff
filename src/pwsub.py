import subprocess

def execute_command(command):
    print("Executing command:", " ".join(command))
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Command failed with exit code:", result.returncode)
        print("Error output:", result.stderr)

def main():
    game = "diffPD"
    opponents = ["NL", "LOLA"]
    for opponent in opponents:
        command = [
            "python",
            "src/main_mfos_ppo_arunim.py",
            f"--game={game}",
            f"--opponent={opponent}",
            f"--exp-name=mfos_ppo_{game}_{opponent}_pwlinear_ccdr",
            "--ccdr",
            "--pwlinear",
        ]
        execute_command(command)

        command = [
            "python",
            "src/main_mfos_ppo_arunim.py",
            f"--game={game}",
            f"--opponent={opponent}",
            f"--exp-name=mfos_ppo_{game}_{opponent}_pwlinear",
            "--pwlinear",
        ]
        execute_command(command)


if __name__ == "__main__":
    main()