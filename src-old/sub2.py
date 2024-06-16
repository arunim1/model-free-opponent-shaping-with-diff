import subprocess
from main_non_mfos import main as main_non_mfos

if __name__ == "__main__":
    for n_neurons in [4, 8, 16, 32, 40]:
        command = ["python", "src/main_non_mfos.py", f"--exp-name=non_mfos_adam_lr_esv_3by{n_neurons}"]
        print("Executing command:", " ".join(command))
        subprocess.run(command, capture_output=False, text=True)

    command = ["gcloud", "compute", "instances", "stop", "instance-arunim", "--zone=us-east1-b"]
    print("Executing command:", " ".join(command))
    subprocess.run(command, capture_output=False, text=True)