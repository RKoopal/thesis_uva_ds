from pathlib import Path
from transformers import AutoTokenizer

from llm_research.inference import run_inference

if __name__ == "__main__":
    """
    Problem: generation does not stop
    """

    # Configuration
    dataset_path = Path("data/xquad_en_validation.json")
    # dataset_path = Path("data/squad_en_train.json")
    # model_id = "xquad_experiment_long/checkpoint-4866"   # "facebook/opt-125m"
    model_id = "bloom_experiment_TrainingType.LORA/checkpoint-5000"  # "facebook/opt-125m"
    # model_id = "facebook/opt-125m"
    is_peft = True
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    batch_size = 1

    # TODO: transfer this config?
    generation_config = {
        # "max_length": 2048,
        # "truncate": True,
        "truncation": True,
        "max_new_tokens": 30,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "eos_token_id": tokenizer.eos_token_id,
    }
    out_path = Path("BLOOM_LORA_EN_PREDICTIONS.json")
    device = "cpu"

    # print the path

    # Run inference
    run_inference(
        is_qlora=False,
        dataset_path=dataset_path,
        model_id=model_id,
        is_peft=is_peft,
        batch_size=batch_size,
        generation_config=generation_config,
        out_path=out_path,
    )
