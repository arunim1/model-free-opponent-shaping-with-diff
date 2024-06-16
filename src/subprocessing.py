import subprocess
import os

# for every folder that starts with non_mfos_pd_widelrs_diffabs_G
# folders = [f for f in os.listdir(".") if f.startswith("non_mfos_pd_widelrs_diffabs_G")]
# for folder in folders:
#     command = [
#         "python",
#         "src/plot/lresv.py",
#         f"--folder={folder}",
#     ]
#     subprocess.run(command, capture_output=False, text=True)

for G in [1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]:
    command = [
        "python",
        "src/main_non_mfos.py",
        f"--exp-name=non_mfos_pd_widelrs_diffabs_G{G}",
        f"--G={G}",
    ]
    print("Executing command:", " ".join(command))
    subprocess.run(command, capture_output=False, text=True)
