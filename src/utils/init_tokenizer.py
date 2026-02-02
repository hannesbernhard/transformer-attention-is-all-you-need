from pathlib import Path
from transformers import GPT2Tokenizer
from config.paths import GPT2_FROM_BPE


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

