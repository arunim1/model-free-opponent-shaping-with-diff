import subprocess
import os

for G in [2, 2.25, 2.5, 2.75, 3]:
    for pwlinear in [4, 8, 16, 32]:
        command = [
            "python",
            "src/main_non_mfos.py",
            f"--exp-name=non_mfos_pd_highlrs_diffpw{pwlinear}_G{G}",
            f"--G={G}",
            f"--pwlinear={pwlinear}",
        ]
        print("Executing command:", " ".join(command))
        subprocess.run(command, capture_output=False, text=True)

folders = [
    f
    for f in os.listdir(".")
    if (f.startswith("non_mfos_pd_highlrs_diffpw") and not f.endswith("_adam"))
]

for folder in folders:
    command = [
        "python",
        "src/plot/lresv.py",
        f"--folder={folder}",
    ]
    subprocess.run(command, capture_output=False, text=True)
    command = [
        "python",
        "src/plot/pwplots.py",
        f"--folder={folder}",
    ]
    subprocess.run(command, capture_output=False, text=True)
