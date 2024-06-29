import os
from pathlib import Path
from databricks.sdk.runtime import dbutils 

from src.llm_research.experiment_runner import ExperimentArgs, run_experiment
from src.llm_research.train import TrainingType
from Reimer.constants import EVAL_LANGUAGES, MODEL_MAPPING, TRAINING_TYPES

class Args:
    """ Class to hold the arguments as attributes """
    def __init__(self):
        self.pretrained_model = 'mt0-large'
        self.task = 'mix2'
        self.language = 'ar'
        self.data_dir = "data"
        self.output_dir = "output"
        self.type = "ft"
        self.custom_dirs = True
        self.num_train_epochs = 20
        self.gradient_accumulation_steps = 1
        self.per_device_train_batch_size = 16
        self.learning_rate = 1e-4
        self.max_seq_length = 512
        self.save_steps = 200
        self.eval_steps = 200
        self.filter_too_long_samples = True
        self.pad_token_to_eos = False
        self.debug = False
        self.debug_sample_size = 100
        self.token_level_evaluation = True

# Manually set up args
cmd_args = Args()

# Calculate paths based on arguments
if cmd_args.custom_dirs:
    input_dir = Path("/dbfs/Reimer") / cmd_args.data_dir / cmd_args.task
    output_dir = Path("/dbfs/Reimer/output") / cmd_args.task / f"experiment_{cmd_args.pretrained_model}_{cmd_args.task}_{cmd_args.type}_{cmd_args.language}"
else:
    input_dir = Path(cmd_args.data_dir)
    output_dir = Path(cmd_args.output_dir) / f"experiment_{cmd_args.pretrained_model}_{cmd_args.task}_{cmd_args.language}"

#create dbfs dirs
dbutils.fs.mkdirs(str(output_dir))

train_path = input_dir / f"{cmd_args.task}_{cmd_args.language}_train.json"
val_path = input_dir / f"{cmd_args.task}_{cmd_args.language}_validation.json"
# test_paths = [input_dir / f"{cmd_args.task}_{lang}_validation.json" for lang in EVAL_LANGUAGES[cmd_args.task]]
test_paths = []


# Create ExperimentArgs instance directly with all required fields
args = ExperimentArgs(
    pretrained_model_id=MODEL_MAPPING[cmd_args.pretrained_model],
    task_name=cmd_args.task,
    language=cmd_args.language,
    dataset_train_path=str(train_path),
    dataset_val_path=str(val_path),
    output_dir=str(output_dir),
    num_train_epochs=cmd_args.num_train_epochs,
    gradient_accumulation_steps=cmd_args.gradient_accumulation_steps,
    per_device_train_batch_size=cmd_args.per_device_train_batch_size,
    learning_rate=cmd_args.learning_rate,
    max_seq_length=cmd_args.max_seq_length,
    save_steps=cmd_args.save_steps,
    eval_steps=cmd_args.eval_steps,
    filter_too_long_samples=cmd_args.filter_too_long_samples,
    pad_token_to_eos=cmd_args.pad_token_to_eos,
    debug=cmd_args.debug,
    debug_sample_size=cmd_args.debug_sample_size,
    token_level_evaluation=cmd_args.token_level_evaluation,
    dataset_test_paths=test_paths,
    training_type=cmd_args.type
)

# Run experiment
run_experiment(args=args)
