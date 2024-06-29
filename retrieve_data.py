import os 
import json
from pathlib import Path
from datasets import load_dataset

from constants import (
    DATA_DIR,
    TASKS,
    LANG_MAPPING,
    SPLITS,
    INSTRUCTION_FUNCTIONS,
    RETRIEVAL_LANGS_TEST,
    RETRIEVAL_LANGS_TRAIN,
)

SKIP_TASKS = [
    "sib200",
    # "xnli",
    "wikiann"
]

def get_save_data(
    task,
    language,
    save_dir
):
    print(f"Retrieving for {task}-{language}")
    data = load_dataset(TASKS[task], language)

    for split in SPLITS:
        if language in LANG_MAPPING.keys():
            language = LANG_MAPPING[language]
        data_name = f"{task}_{language}_{split}.json"
        print(f"Data name: {data_name}")
        out_path = save_dir / data_name
        # Call dataset creation function
        INSTRUCTION_FUNCTIONS[task](dataset_partition=data[split], out_path=out_path)
    

def main():
    for task in TASKS.keys():
        if task in SKIP_TASKS:
            print(f"Skipping {task}")
            continue
        save_dir = Path(DATA_DIR) / task
        print(f"Training languages: {RETRIEVAL_LANGS_TRAIN[task]}")
        for language in RETRIEVAL_LANGS_TRAIN[task]:
            get_save_data(task, language, save_dir)
        print(f"Additional languages: {RETRIEVAL_LANGS_TEST[task]}")
        for language in RETRIEVAL_LANGS_TEST[task]:
            get_save_data(task, language, save_dir)

if __name__ == "__main__":
    main()