
''' 
From 'evaluation/exact_match.py'. 
This function was used to calculate the loss at a token level for the WikiANN task.
'''
def calculate_token_level_match_accuracy(ground_truth_file: Path, prediction_file: Path) -> float:
    """
    Calculate the accuracy on an output token level between ground truth and predictions. Can be used for NER tasks for example.

    Parameters:
    - ground_truth_file (Path): The path to the JSON file containing the ground truth.
    - prediction_file (Path): The path to the JSON file containing the predictions.

    Returns:
    - float: The exact match accuracy as a decimal fraction.

    Raises:
    - ValueError: If the number of entries in the ground truth and prediction files do not match.
    """
    ground_truth_data = load_json(ground_truth_file)
    prediction_data = load_json(prediction_file)

    if len(ground_truth_data) != len(prediction_data):
        raise ValueError(
            "The number of entries in the ground truth and prediction files do not match."
        )
   
    sample_accuracies = []
    mismatch_count = 0
    translation_table = str.maketrans({key: None for key in "[]',"})
    for gt_sample, pred_sample in zip(ground_truth_data, prediction_data):
        gt = gt_sample['answer']
        # Improved conversion of pred to list
        pred = pred_sample['generated_text'].translate(translation_table).strip().split()

        if len(gt) != len(pred):
            sample_accuracies.append(0)
            mismatch_count += 1
        else:
            # Using generator expression for memory efficiency
            accuracy = sum(1 for gt_token, pred_token in zip(gt, pred) if gt_token == pred_token) / len(gt)
            sample_accuracies.append(accuracy)

    overall_accuracy = sum(sample_accuracies) / len(sample_accuracies)
    mismatch_rate = mismatch_count / len(sample_accuracies)

    print(f"Got {mismatch_count} samples with a mismatched length.")
    return (overall_accuracy, mismatch_rate)



'''
From inference/infer_dataset.py.
This function was used to run inference on a dataset using a model without loading the model separately for each task.
Function was called from 'databricks/run_inference.ipynb'
'''
def run_inference_manual(
    model,
    tokenizer,
    model_id,
    dataset_path: Path,
    is_peft: bool,
    is_qlora: bool,
    batch_size: int,
    generation_config: Dict,
    out_path: Path,
    dataset_split: str = "train",
    sample_size: int = None,
    device="cuda",
    is_encoder_decoder = True,
):
    '''
    Version of run_inference where the model is passed rather than reloaded.
    '''
    logger.info(f"Running inference on {dataset_path} using model {model_id}")
    # Validate arguments
    if not isinstance(dataset_path, Path):
        raise TypeError("dataset_path must be a Path object")
    if not isinstance(out_path, Path):
        raise TypeError("out_path must be a Path object")

    # Load the dataset
    dataset = load_dataset("json", data_files=str(dataset_path))
    dataset = dataset[dataset_split]

    if sample_size is not None:
        print(f"Using only {sample_size} samples from the dataset.")
        dataset = dataset.select(range(sample_size))

    if device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model.eval()

    if not is_qlora:
        model.to(device)

        # Initialize stopping criteria - stop at new line char
    stop_criterium = StoppingCriteriaList(
        [
            StopAtLineEndCriterion(tokenizer=tokenizer),
        ]
    )

    # Run inference
    if is_encoder_decoder:
        outputs = infer_model_generate(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            generation_config=generation_config,
            stop_criterion=stop_criterium,
            device=device,
        )
    else:
        outputs = pipeline_inference(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            generation_config=generation_config,
            stop_criterium=stop_criterium,
            device=device,
            is_qlora=is_qlora,
            batch_size=batch_size,
        )

    # Save the outputs to a file
    with out_path.open("wt", encoding="utf-8") as file:
        json.dump(outputs, file)

    logging.info("Inference completed successfully.")


'''
From 'data_prep/prompt_templates.py'.
Below are the prompt templates used for formatting the data for mT0.
'''
from jinja2 import Template
XNLI_ZERO_SHOT_PROMPT_TEMPLATE = Template(
"""
Given the following: "{{ premise }}"
Then the following statement: "{{ hypothesis }}" can be labeled as "entailment", "neutral", or "contradiction"?
"""
)
XNLI_ZERO_SHOT_TARGET_TEMPLATE = Template("""{{ answer_choices[label] }}""")
XNLI_ANSWER_CHOICES = {0: "entailment", 1: "neutral", 2: "contradiction"} 


SIB200_ZERO_SHOT_TEMPLATE = Template(
""" Given the topics of {{answer_choices[:-1] | join(', ') }}, and {{
answer_choices[-1] }}, specify which of them best represents the following sentence:
"{{ text }}"
Answer:
"""
)
SIB200_ZERO_SHOT_TARGET_TEMPLATE = Template("""{{ category }}""")
SIB200_ANSWER_CHOICES = ["science/technology", "geography", "entertainment", "politics", "health", "travel", "sports"]


WIKIANN_ZERO_SHOT_TEMPLATE = Template(
    """
    Classify each word in "{{ tokens }}", as either "person", "location", "organisation" or "none".
    Example:
    Text: ["John", "Doe", "from", "Microsoft", "visits", "Berlin"]
    Classified: ["person", "person", "none", "organisation", "none", "location"]

    Classified:
    """
)
WIKIANN_ZERO_SHOT_TARGET_TEMPLATE = Template("""{{ answer_choices[tag] }}""")
WIKIANN_ANSWER_CHOICES = {0: "none", 1 : "person", 2 : "person", 3 : "organisation", 4 : "organisation", 5 : "location" , 6 : "location"}