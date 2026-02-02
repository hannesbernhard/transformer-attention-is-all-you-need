import csv
import uuid
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.modelling.model.transformer import TransformerConfig, TransformerModel
from src.utils.data_cleaning import clean_dataset
from src.dataset import TranslationDataset
import yaml
from tqdm import tqdm
import logging
from config.paths import EVALUATION, GPT2_FROM_BPE, MODEL_CONFIG, BEST_MODELS


class TransformerEvaluator:
    def __init__(self, model, tokenizer, test_dataset, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.beam_width = 3
        self.logger = logging.getLogger(__name__)

    @torch.no_grad()
    def greedy_decode(self, source_ids, source_mask, max_len=64):
        batch_size = source_ids.size(0)

        # Initialize with BOS token
        decoder_input = (
            torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
            * self.tokenizer.bos_token_id
        )

        for _ in range(max_len - 1):
            # Get model output (the Attention module will handle causal masking)
            output = self.model(source_ids, decoder_input, source_mask, None)

            # Get next token predictions
            next_token_logits = output[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)

            # Concatenate with previous output
            decoder_input = torch.cat([decoder_input, next_tokens], dim=-1)

            # Check if all sequences have EOS token
            if all((decoder_input == self.tokenizer.eos_token_id).any(dim=1)):
                break

        return decoder_input

    @torch.no_grad()
    def beam_search_decode(self, source_ids, source_mask, max_len=64):
        batch_size = source_ids.size(0)

        # Initialize with BOS token
        decoder_input = (
            torch.ones((batch_size, 1), dtype=torch.long, device=self.device)
            * self.tokenizer.bos_token_id
        )

        # Initialize beams (each entry is a tuple: (sequence, score))
        sequences = [
            (decoder_input, torch.zeros(batch_size, device=self.device))
        ]  # Initially BOS token with score 0

        for _ in range(max_len - 1):
            all_candidates = []

            # Expand each sequence in the beam
            for seq, score in sequences:
                # Get model output (the Attention module will handle causal masking)
                output = self.model(source_ids, seq, source_mask, None)

                # Get the next token logits
                next_token_logits = output[:, -1, :]
                next_token_probs = torch.nn.functional.softmax(
                    next_token_logits, dim=-1
                )

                # Get the top `beam_width` tokens
                topk_scores, topk_tokens = next_token_probs.topk(
                    self.beam_width, dim=-1
                )

                # Add new candidates to the list
                for i in range(self.beam_width):
                    next_token = topk_tokens[:, i].unsqueeze(1)
                    new_score = score - torch.log(
                        topk_scores[:, i] + 1e-10
                    )  # Using negative log for the score
                    new_seq = torch.cat([seq, next_token], dim=-1)
                    all_candidates.append((new_seq, new_score))

            # Select the `beam_width` best candidates by sorting based on the score
            # We need to flatten the batch and beam dimensions for sorting
            all_candidates = sorted(
                all_candidates, key=lambda x: x[1].sum().item()
            )  # Sum scores across batch
            sequences = all_candidates[: self.beam_width]

            # Check if all sequences have EOS token
            if all((seq == self.tokenizer.eos_token_id).any() for seq, _ in sequences):
                break

        # Return the sequence with the best score
        return sequences[0][0]

    @torch.no_grad()
    def beam_search_decode_2(self, source_ids, source_mask, max_len=64):
        batch_size = source_ids.size(0)
        device = source_ids.device

        # Initialize with BOS token
        decoder_input = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )

        # Initialize beams (each entry is a tuple: (sequence, score, finished))
        sequences = [
            (
                decoder_input,
                torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            )
        ]

        for _ in range(max_len - 1):
            all_candidates = []

            # Expand each sequence in the beam
            for seq, score, finished in sequences:
                if finished.all():
                    all_candidates.append((seq, score, finished))
                    continue

                # Get model output
                output = self.model(source_ids, seq, source_mask, None)

                # Get the next token logits
                next_token_logits = output[:, -1, :]
                next_token_probs = torch.nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )

                # Get the top `beam_width` tokens
                topk_log_probs, topk_tokens = next_token_probs.topk(
                    self.beam_width, dim=-1
                )

                # Add new candidates to the list
                for i in range(self.beam_width):
                    next_token = topk_tokens[:, i].unsqueeze(1)
                    new_score = score - topk_log_probs[:, i]
                    new_seq = torch.cat([seq, next_token], dim=-1)
                    new_finished = finished | (
                        next_token.squeeze(1) == self.tokenizer.eos_token_id
                    )
                    all_candidates.append((new_seq, new_score, new_finished))

            # Select the `beam_width` best candidates
            all_candidates.sort(key=lambda x: x[1].sum().item())
            sequences = all_candidates[: self.beam_width]

            # Check if all sequences have finished
            if all(finished.all() for _, _, finished in sequences):
                break

        # Select the best sequence
        best_seq = max(sequences, key=lambda x: x[1].sum().item())[0]
        return best_seq

    @torch.no_grad()
    def beam_search_decode_3(self, source_ids, source_mask, max_len=64):
        device = source_ids.device
        batch_size = source_ids.size(0)

        # Initialize beams: (sequence, score, finished)
        beams = [
            (
                torch.full(
                    (batch_size, 1),
                    self.tokenizer.bos_token_id,
                    dtype=torch.long,
                    device=device,
                ),
                torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            )
        ]

        for _ in range(max_len - 1):
            candidates = []
            for seq, score, finished in beams:
                if finished.all():
                    candidates.append((seq, score, finished))
                    continue

                output = self.model(source_ids, seq, source_mask, None)
                next_token_logits = output[:, -1, :]
                next_token_probs = torch.nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )
                topk_probs, topk_tokens = next_token_probs.topk(self.beam_width, dim=-1)

                for i in range(self.beam_width):
                    new_seq = torch.cat([seq, topk_tokens[:, i].unsqueeze(1)], dim=-1)
                    new_score = score - topk_probs[:, i]
                    new_finished = finished | (
                        topk_tokens[:, i] == self.tokenizer.eos_token_id
                    )
                    candidates.append((new_seq, new_score, new_finished))

            # Select top beams
            beams = sorted(candidates, key=lambda x: x[1].sum().item())[
                : self.beam_width
            ]

            if all(finished.all() for _, _, finished in beams):
                break

        # Select and trim best sequence
        best_seq = max(beams, key=lambda x: x[1].sum().item())[0]
        eos_indices = (best_seq == self.tokenizer.eos_token_id).nonzero()
        return (
            best_seq[:, : eos_indices[0, 1] + 1]
            if eos_indices.numel() > 0
            else best_seq
        )

    def calculate_bleu(self, predictions, references):
        """Calculate BLEU score for batches of predictions and references."""
        smooth_fn = SmoothingFunction().method1

        # Decode predictions and references
        pred_texts = [
            self.tokenizer.decode(pred, skip_special_tokens=True)
            for pred in predictions
        ]
        ref_texts = [
            self.tokenizer.decode(ref, skip_special_tokens=True) for ref in references
        ]

        # Calculate BLEU scores
        bleu_scores = [
            sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
            for pred, ref in zip(pred_texts, ref_texts)
        ]

        return bleu_scores, pred_texts, ref_texts

    @torch.no_grad()
    def evaluate(self, max_len=64, decoding_strategy="greedy"):
        """Evaluate model and save results to CSV."""
        self.model.eval()
        all_bleu_scores = []
        examples = []
        highest_bleu = 0

        # 2 digit uid
        uid = str(uuid.uuid4())[:2]
        csv_path = EVALUATION / f"{uid}_evaluation_results_{decoding_strategy}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "prediction", "reference", "bleu"])

            for batch in tqdm(self.test_loader):
                # Generate predictions
                source_ids = batch["source_ids"].to(self.device)
                source_mask = batch["source_mask"].to(self.device)
                target_ids = batch["labels"].to(self.device)

                # Choose the appropriate decoding strategy
                if decoding_strategy == "greedy":
                    predictions = self.greedy_decode(source_ids, source_mask, max_len)
                elif decoding_strategy == "beam_search":
                    predictions = self.beam_search_decode_3(
                        source_ids, source_mask, max_len
                    )
                else:
                    raise ValueError(f"Invalid decoding strategy: {decoding_strategy}")

                bleu_scores, pred_texts, ref_texts = self.calculate_bleu(
                    predictions, target_ids
                )

                # Save results
                for src, pred, ref, bleu in zip(
                    batch["source"], pred_texts, ref_texts, bleu_scores
                ):
                    writer.writerow([src, pred, ref, round(bleu, 4)])
                    all_bleu_scores.append(bleu)
                    if bleu > highest_bleu:
                        highest_bleu = bleu
                        print(f"New highest BLEU score: {highest_bleu}")
                        print(f"Source: {src} \nPrediction: {pred} \nReference: {ref}")

        avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
        return avg_bleu, examples


def load_model(model_path, config):
    """Load the trained model."""
    model = TransformerModel(config=TransformerConfig(**config))
    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main():
    # Load configurations
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)

    # Initialize tokenizer
    max_length = model_config.get("max_len", 64)
    tokenizer = GPT2Tokenizer.from_pretrained(str(GPT2_FROM_BPE / str(max_length)))
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "unk_token": "[UNK]",
        }
    )

    # Load and prepare test dataset
    test_ds = load_dataset(
        "wmt17", "de-en", split="test"
    )  # Test set only contains 3000 samples
    test_cleaned = clean_dataset(test_ds, max_length)
    test_dataset = TranslationDataset(test_cleaned, tokenizer, max_length=max_length)

    print(f"Test dataset size: {len(test_dataset)}")

    # Load model
    model = load_model(BEST_MODELS / "best_model.pth", model_config)

    # Create evaluator and run evaluation
    evaluator = TransformerEvaluator(model, tokenizer, test_dataset, batch_size=128)
    bleu_score, examples = evaluator.evaluate(
        max_len=max_length, decoding_strategy="beam_search"
    )

    # Print results
    print(f"\nAverage BLEU Score: {bleu_score:.4f}")
    print("\nExample Translations:")
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Source:     {example['source']}")
        print(f"Prediction: {example['prediction']}")
        print(f"Reference:  {example['reference']}")
        print(f"BLEU Score: {example['bleu']:.4f}")


if __name__ == "__main__":
    main()
