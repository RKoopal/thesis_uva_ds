import os
import argparse
from pathlib import Path
from llm_research.experiment_runner import ExperimentArgs, run_experiment
from llm_research.train import TrainingType



from constants import (
    EVAL_LANGUAGES,
    MODEL_MAPPING,
    TRAINING_TYPES
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a machine learning experiment.")
    
    # Define default values as constants,
    DEFAULT_MODEL = None
    DEFAULT_TASK = None
    DEFAULT_LANGUAGE = None
    DEFAULT_DATA_DIR = "data"
    DEFAULT_OUTPUT_DIR = "output"
    DEFAULT_TYPE = "ft"
    DEFAULT_CUSTOM_DIR = True
    DEFAULT_NUM_TRAIN_EPOCHS = 1
    DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
    DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_MAX_SEQ_LENGTH = 512
    DEFAULT_SAVE_STEPS = 500    #500
    DEFAULT_EVAL_STEPS = 500    #500
    DEFAULT_FILTER_TOO_LONG_SAMPLES = True
    DEFAULT_PAD_TOKEN_TO_EOS = False
    DEFAULT_DEBUG = False
    DEFAULT_DEBUG_SAMPLE_SIZE = 100
    DEFAULT_TOKEN_LEVEL_EVALUATION = False           # set true for wikiann

    # Add arguments with the default values
    parser.add_argument("-pt", "--pretrained_model", type=str, default=DEFAULT_MODEL, help="Pretrained model identifier")
    parser.add_argument("-t", "--task", type=str, default=DEFAULT_TASK, help="Task identifier")
    parser.add_argument("-l", "--language", type=str, default=DEFAULT_LANGUAGE, help="Language identifier")
    parser.add_argument("-d", "--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Data directory (default: data)")
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory (default: output)")
    parser.add_argument("--type", type=str, default=DEFAULT_TYPE, help="Training type")
    parser.add_argument("--custom_dirs", type=str, default=DEFAULT_CUSTOM_DIR, help="Output directory (default: output)")
    parser.add_argument("--num_train_epochs", type=int, default=DEFAULT_NUM_TRAIN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE, help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_EVAL_STEPS, help="Evaluation steps")
    parser.add_argument("--filter_too_long_samples", action='store_true', default=DEFAULT_FILTER_TOO_LONG_SAMPLES, help="Filter out samples that are too long")
    parser.add_argument("--pad_token_to_eos", action='store_true', default=DEFAULT_PAD_TOKEN_TO_EOS, help="Pad token to the end of the sequence")
    parser.add_argument("--debug", action='store_true', default=DEFAULT_DEBUG, help="Enable debug mode")
    parser.add_argument("--debug_sample_size", type=int, default=DEFAULT_DEBUG_SAMPLE_SIZE, help="Debug sample size")
    parser.add_argument("--token_level_evaluation", type=bool, default=DEFAULT_TOKEN_LEVEL_EVALUATION, help="Debug sample size")

    return parser.parse_args()


if __name__ == "__main__":
    # Configuration
    cmd_args = parse_args()

    if not (cmd_args.pretrained_model and cmd_args.task and cmd_args.language):
        raise ValueError(f"Not all essential cmd's present, specify --pretrained_model (-pt), --task (-t) and (--language) -l")

    if cmd_args.custom_dirs:
        input_dir = Path(cmd_args.data_dir) / cmd_args.task
        output_dir = Path(cmd_args.output_dir) / cmd_args.task / f"experiment_{cmd_args.pretrained_model}_{cmd_args.task}_{cmd_args.type}_{cmd_args.language}"
    else:
        input_dir = Path(cmd_args.data_dir)
        output_dir = Path(cmd_args.output_dir) / f"experiment_{cmd_args.pretrained_model}_{cmd_args.task}_{cmd_args.language}"

    train_path = input_dir / f"{cmd_args.task}_{cmd_args.language}_train.json"
    val_path = input_dir / f"{cmd_args.task}_{cmd_args.language}_validation.json"
    test_paths = [input_dir /  f"{cmd_args.task}_{lang}_validation.json" for lang in EVAL_LANGUAGES[cmd_args.task]]

    local_args = {
        "pretrained_model_id": MODEL_MAPPING[cmd_args.pretrained_model],
        "task_name": cmd_args.task,
        "language": cmd_args.language,
        "dataset_train_path": train_path,       
        "dataset_val_path": val_path,
        "output_dir": output_dir,
        "num_train_epochs": cmd_args.num_train_epochs,
        "gradient_accumulation_steps": cmd_args.gradient_accumulation_steps,
        "per_device_train_batch_size": cmd_args.per_device_train_batch_size,
        "learning_rate": cmd_args.learning_rate,
        "max_seq_length": cmd_args.max_seq_length,
        "save_steps": cmd_args.save_steps,
        "eval_steps": cmd_args.eval_steps,
        "filter_too_long_samples": cmd_args.filter_too_long_samples,
        "pad_token_to_eos": cmd_args.pad_token_to_eos,
        "debug": cmd_args.debug,
        "debug_sample_size": cmd_args.debug_sample_size,
        "token_level_evaluation" : cmd_args.token_level_evaluation
    }

    for k, v in local_args.items():
        if os.environ.get(k) is None:
            os.environ[k] = str(v)

    args = ExperimentArgs()

    args.dataset_test_paths = test_paths

    args.training_type = TRAINING_TYPES[cmd_args.type]

    args.language = cmd_args.language

    if cmd_args.task == 'wikiann':
        args.token_level_evaluation = True


    # Run experiment
    run_experiment(args=args)
