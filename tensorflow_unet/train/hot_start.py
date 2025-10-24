# create a python script which restarts the program 'bin/python train.py' and if the program ends, and the last row of the output isn't "finished", restart the 
import subprocess
import time
import subprocess
import os
import signal


def run_and_monitor():
    while True:
        print("Starting training script...")
        process = subprocess.Popen(
            ["bin/python", "train/dynamic_quant_train.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        last_line = ""
        for line in process.stdout:
            # Strip and store last line
            stripped_line = line.strip()
            last_line = stripped_line

            # Check for escape char (e.g., "\x1b") and filter
            # and "val_accuracy" not in line:

          
            print(line, end='')  # Print allowed output


        process.wait()
        print(f"Process exited with code {process.returncode}")

        if last_line.lower() != "finished":
            print("Last line was not 'finished'. Restarting...")
            kill_gpu_process_by_name("bin/python")
            time.sleep(10)  # Optional: brief pause before restart
            kill_gpu_process_by_name("bin/python")
            time.sleep(1)  # Optional: brief pause before restart
            kill_gpu_process_by_name("bin/python")
            time.sleep(1)  # Optional: brief pause before restart
            kill_gpu_process_by_name("bin/python")
            time.sleep(1)  # Optional: brief pause before restart
            kill_gpu_process_by_name("bin/python")

        else:
            print("Training finished successfully.")
            break

def kill_gpu_process_by_name(target_name):
    try:
        # Run `nvidia-smi` and get list of running processes
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check if command ran successfully
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return

        if len(result.stdout) == 0:
            print(f"Process not found: {target_name}")
            return

        # Search for matching process name
        for line in result.stdout.strip().split("\n"):
            pid_str, name = [item.strip() for item in line.split(',')]
            if name == target_name:
                print(f"Killing process {name} with PID {pid_str}")
                os.kill(int(pid_str), signal.SIGKILL)  # same as kill -9
                return

        print(f"No matching process named '{target_name}' found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    kill_gpu_process_by_name("bin/python")
    time.sleep(1)  # Optional: brief pause before restart
    kill_gpu_process_by_name("bin/python")
    time.sleep(1)  # Optional: brief pause before restart
    kill_gpu_process_by_name("bin/python")
    time.sleep(1)  # Optional: brief pause before restart
    kill_gpu_process_by_name("bin/python")

    run_and_monitor()