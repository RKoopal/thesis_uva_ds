from llm_research.data_prep import (
    create_sib200_instruction_dataset, 
    create_xnli_zero_shot_instruction_dataset,
    create_wikiann_zero_shot_instruction_dataset,
    create_sib200_zero_shot_instruction_dataset
)

from llm_research.train import TrainingType

DATA_DIR = "data"

TASKS = {
    "sib200": "Davlan/sib200",
    "xnli" : "xnli",
    "wikiann" : "wikiann"
}

RETRIEVAL_LANGS_TRAIN = {
    "sib200" : ["deu_Latn", "arb_Arab", "spa_Latn", "ell_Grek"],
    "xnli": ["de","ar","es","el"],
    "wikiann": ["de","ar","es","el"]
}

RETRIEVAL_LANGS_TEST = {
    "sib200" : ["fra_Latn", "rus_Cyrl"],
    "xnli": ["fr","ru"],
    "wikiann": ["fr","ru"]
}

TRAINING_LANGUAGES = {
    "sib200" : ["de","ar","es","el"],
    "xnli": ["de","ar","es","el"],
    "wikiann": ["de","ar","es","el"]
}

# additional eval languages
ADD_LANGUAGES = {
    "sib200" : ["fr", "ru"],
    "xnli": ["fr","ru"],
    "wikiann": ["fr","ru"]
}



EVAL_LANGUAGES = {
    "sib200" : TRAINING_LANGUAGES['sib200'] + ADD_LANGUAGES['sib200'],
    "xnli":  TRAINING_LANGUAGES['xnli'] + ADD_LANGUAGES['xnli'],
    "wikiann":  TRAINING_LANGUAGES['wikiann'] + ADD_LANGUAGES['wikiann']
}

SPLITS = [
    "train",
    "validation"
]

INSTRUCTION_FUNCTIONS = {
    "sib200": create_sib200_zero_shot_instruction_dataset,
    "xnli" : create_xnli_zero_shot_instruction_dataset,
    "wikiann" : create_wikiann_zero_shot_instruction_dataset
}

#acceptables values
TRAINING_TYPES = {
    "qlora": TrainingType.QLORA,
    "p_tune": TrainingType.P_TUNE,
    "ft": TrainingType.FULL_FINETUNE,
    "lora": TrainingType.LORA,
}

# mappings path friendly id to huggingface id
MODEL_MAPPING = {
    "mt5-small" : "google/mt5-small",
    "mt0-small" : "bigscience/mt0-small",
    "mt0-large" : "bigscience/mt0-large",
    "bloom-560m" : "bigscience/bloom-560m"
}

LANG_MAPPING = {
    "deu_Latn" : "de",
    "arb_Arab" : "ar",
    "spa_Latn" : "es",
    "ell_Grek": "el",
    "fra_Latn" : "fr",
    "rus_Cyrl" : "ru"
}