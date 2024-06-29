import os
from datetime import datetime
import subprocess


from constants import (
    TRAINING_LANGUAGES
)

SKIP_TASKS = [
    "wikiann",
    "xnli",
    # "sib200"
]


def run_experiments():

    script = "run_baseline_exp.py"
    model = "mt0-large"
    type_ = "lora"

    logging_file = f"experiment_results_{model}_{type_}_sib200_all_langs.txt"

    with open(logging_file, "w", encoding="utf-8") as f:
        for task, languages in TRAINING_LANGUAGES.items():
            if task in SKIP_TASKS:
                print(f"skipping {task}")
                continue
            for language in languages:
                print(f"Running {model} for {task} and {language}")
                command = ["python", f"{script}", "-pt", f"{model}", "-t", f"{task}", "-l", f"{language}", "--type", f"{type_}"]
                try:
                    result = subprocess.run(command, capture_output=True, text=True)
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write("------------- NEW EXPERIMENT -------------")
                    f.write(f"[{current_time}] For {task} - {language}:\n")
                    f.write(f"Returncode = {result.returncode}\n")
                    f.write(result.stdout)

                    if result.returncode != 0:
                        f.write(f"Error: {result.stderr}\n")

                    # Ensures the content is written to the file immediately
                    f.flush()
                except Exception as e:
                    f.write(f"Failed to run command {command}: {e}\n")
                    f.flush()

                # Optionally, print the command and result to console as well
                print(f"[{current_time}] Command: {' '.join(command)}")
                print(f"Returncode: {result.returncode}")
                if result.returncode == 0:
                    print("Output:", result.stdout)
                else:
                    print("Error:", result.stderr)


if __name__ == "__main__":
    run_experiments()