from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer
from config.paths import GPT2_FROM_BPE, MODEL_CONFIG
import yaml
from src.utils.data_cleaning import clean_dataset
from src.modelling.embedding.tokenizer import ByteLevelBPETokenizer


def get_or_create_tokenizer() -> GPT2Tokenizer:
    """
    Load a previously trained GPT2-compatible tokenizer.
    Assumes tokenizer files already exist (--> copied from Clolab).
    """

    vocab_path = Path(GPT2_FROM_BPE) / "vocab.json"
    merges_path = Path(GPT2_FROM_BPE) / "merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        raise RuntimeError(
            "Tokenizer files not found. "
            "Train tokenizer in Colab and copy cache locally."
        )

    tokenizer = GPT2Tokenizer.from_pretrained(
        str(GPT2_FROM_BPE),
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        add_prefix_space=True,
    )
    return tokenizer



# Load the dataset
def init_tokenizer(percentage=1):
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)
    dataset = load_dataset("wmt17", "de-en", split=f"train[:{percentage}%]")
    cleaned_data = clean_dataset(dataset, max_length=model_config["max_len"])
    ByteLevelBPETokenizer(
        cleaned_data,
        vocab_size=model_config["vocab_size"],
        max_length=model_config["max_len"],
    )



