import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import yaml
import os
from pathlib import Path
import argparse

from src.modelling.model.transformer import TransformerConfig, TransformerModel
from src.dataset import TranslationDataset
from src.utils.data_cleaning import clean_dataset
from src.utils.init_tokenizer import get_or_create_tokenizer
from config.paths import MODEL_CONFIG, BEST_MODELS

project_root = Path(__file__).parent.parent.absolute()
DATASET_PATH = os.path.join(
    project_root,
    "hf_cache",
    "root",
    ".cache",
    "huggingface",
    "datasets",
    "wmt17",
    "de-en",
    "0.0.0",
    "54d3aacfb5429020b9b85b170a677e4bc92f2449",
)


@torch.no_grad()
def evaluate(model, tokenizer, dataloader, device, max_len):
    model.eval()
    smooth = SmoothingFunction().method1

    bleu_scores = []
    printed = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        source_ids = batch["source_ids"].to(device)
        labels = batch["labels"]

        # batch_size = 1 is safest for generate()
        for i in range(source_ids.size(0)):
            generated = model.generate(
                source_ids[i : i + 1],
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_len,
            )

            pred = tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
            ref = tokenizer.decode(
                labels[i], skip_special_tokens=True
            )

            bleu = sentence_bleu(
                [ref.split()], pred.split(), smoothing_function=smooth
            )
            bleu_scores.append(bleu)

            # Print a few examples
            if printed < 5:
                print("\nSOURCE:")
                print(batch["source"][i])
                print("PREDICTION:")
                print(pred)
                print("REFERENCE:")
                print(ref)
                print(f"BLEU: {bleu:.4f}")
                print("-" * 60)
                printed += 1

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== Load model config =====
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)

    max_len = model_config.get("max_len", 64)

    # ===== Load tokenizer =====
    tokenizer = get_or_create_tokenizer()

    # ===== Load model =====
    model = TransformerModel(
        TransformerConfig(**model_config)
    ).to(device)

    checkpoint = torch.load(
        BEST_MODELS / "best_model_without_rope.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"],strict=False)

    # ===== Load SMALL test subset (KEY PART) =====

    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch_data_online", action="store_true")
    parser.add_argument("--max_len", type=int, default = 64)
    args = parser.parse_args()

    if args.fetch_data_online:
        test_ds = load_dataset(
            "wmt17",
            "de-en",
            split="test[:1000]" 
        )
    else:
        test_ds = load_dataset(
            str(DATASET_PATH),
            split=f"train[:1000]",
        )
    test_cleaned = clean_dataset(
        test_ds,
        min_len=5,
        max_len=args.max_len,
        max_ratio=2.5,
    )

    test_dataset = TranslationDataset(
        test_cleaned,
        tokenizer,
        args.max_len,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,      # IMPORTANT for generate()
        shuffle=False,
    )

    print(f"Evaluating on {len(test_dataset)} test sentences")

    bleu = evaluate(
        model,
        tokenizer,
        test_loader,
        device,
        args.max_len,
    )

    print("\n==============================")
    print(f"Average BLEU score: {bleu:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()