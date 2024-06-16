import subprocess

for G in [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]:
    command = [
        "python",
        "src/main_non_mfos.py",
        f"--exp-name=non_mfos_pd_widelrs_G{G}",
        f"--G={G}",
    ]
    subprocess.run(command, capture_output=False, text=True)
