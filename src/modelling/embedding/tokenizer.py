from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import GPT2Tokenizer
import os
import json

from config.paths import GPT2_FROM_BPE


class ByteLevelBPETokenizer:
    def __init__(self, vocab_size=10_000, min_frequency=1, lowercase=False):
        """
        Initializes a byte-level BPE tokenizer configuration.
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        
        # Initialize tokenizer with empty BPE model
        self.tokenizer = Tokenizer(models.BPE())

        # Pre-tokenizer: ByteLevel handles all Unicode safely
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Decoder: ByteLevel merges byte sequences back into text
        self.tokenizer.decoder = decoders.ByteLevel()

        # Post-processing: ensures leading-space token handling (Ġ)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    def train(self, files):
        """
        Train BPE on a list of text files.
        """
        if isinstance(files, str):
            files = [files]

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        )

        self.tokenizer.train(files, trainer)

    def save(self, path):
        """
        Save tokenizer to a JSON file.
        """
        self.tokenizer.save(path, pretty=True)

    def load(self, path):
        """
        Load tokenizer from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        """
        Encode text → token IDs and tokens.
        """
        if self.lowercase:
            text = text.lower()
        return self.tokenizer.encode(text)

    def decode(self, ids):
        """
        Decode token IDs → text.
        """
        return self.tokenizer.decode(ids)

    def export_to_gpt2(self):
        model = self.tokenizer.model
        vocab = model.get_vocab()
        merges = model.get_merges()

        vocab_path = os.path.join(GPT2_FROM_BPE, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            json.dump(vocab, vocab_file, ensure_ascii=False) # Do not escape non-ASCII characters

        merges_path = os.path.join(GPT2_FROM_BPE, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as merges_file:
            merges_file.write("#version: 0.2\n") # expected by GPT2Tokenizer
            merges_file.write(
                "\n".join(" ".join(map(str, merge)) for merge in merges)
            )

        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
            str(GPT2_FROM_BPE),
            pad_token = "[PAD]",
            bos_token = "[BOS]",
            eos_token = "[EOS]",
            unk_token = "[UNK]",
            add_prefix_space = True,
        )

        return gpt2_tokenizer

