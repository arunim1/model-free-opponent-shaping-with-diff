import subprocess
import os

for G in [2, 2.25, 2.5, 2.75, 3]:
    for threshold in ["abs", "exp0", "exp1", "squared"]:
        command = [
            "python",
            "src/main_non_mfos.py",
            f"--exp-name=non_mfos_pd_highlrs_diff{threshold}_G{G}_adam",
            f"--G={G}",
            f"--threshold={threshold}",
        ]
        print("Executing command:", " ".join(command))
        subprocess.run(command, capture_output=False, text=True)

folders = [
    f
    for f in os.listdir(".")
    if (f.startswith("non_mfos_pd_highlrs_diff") and f.endswith("_adam"))
]
for folder in folders:
    command = [
        "python",
        "src/plot/lresv.py",
        f"--folder={folder}",
    ]
    subprocess.run(command, capture_output=False, text=True)
