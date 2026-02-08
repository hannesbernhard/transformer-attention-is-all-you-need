import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
import os

from src.modelling.model.transformer import TransformerModel, TransformerConfig
from src.modelling.embedding.positional_encoding import PositionalEncoding
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
def evaluate(model, model_config, tokenizer, dataloader, device, max_len):
    model.eval()
    smooth = SmoothingFunction().method1
    bleu_scores = []

    for batch in tqdm(dataloader, desc=f"Evaluating (max_len={max_len})"):
        source_ids = batch["source_ids"].to(device)
        labels = batch["labels"]

        generated = model.generate(
            source_ids,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=None,
            max_length=max_len,
        )

        for pred_ids, ref_ids in zip(generated, labels):
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
            ref_ids = ref_ids[:model_config["max_len"]]
            ref = tokenizer.decode(ref_ids, skip_special_tokens=True)

            bleu = sentence_bleu(
                [ref.split()],
                pred.split(),
                smoothing_function=smooth,
            )
            bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores)


def extend_sinusoidal_pe(model, d_model, test_max_len, device):
    """
    Extend positional encodings at evaluation time ONLY.
    This does NOT retrain the model.
    """
    pe = PositionalEncoding(d_model, test_max_len).to(device)

    if model.src_embed.pos_emb is not None:
        model.src_embed.pos_emb = pe

    if model.tgt_embed.pos_emb is not None:
        model.tgt_embed.pos_emb = pe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_max_len", type=int, required=True)
    parser.add_argument("--split", type=str, default="test[:300]")
    parser.add_argument("--fetch_data_online", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Testing with max_len = {args.test_max_len}")

    # ===== Load config =====
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)

    tokenizer = get_or_create_tokenizer()

    # ===== Load model =====
    model = TransformerModel(
        TransformerConfig(**model_config)
    ).to(device)

    checkpoint = torch.load(
        BEST_MODELS / "best_model_with_rope.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict = False)

    # ===== Extend PE ONLY if sinusoidal =====
    if not model_config.get("use_rope", False):
        print("Extending sinusoidal positional encodings for evaluation")
        extend_sinusoidal_pe(
            model,
            d_model=model_config["d_model"],
            test_max_len=args.test_max_len,
            device=device,
        )
    else:
        print("RoPE enabled — no positional encoding extension needed")

    # ===== Load dataset =====
    if args.fetch_data_online:
        dataset = load_dataset(
            "wmt17", "de-en", split = args.split
        )
    else:
        dataset = load_dataset(
            str(DATASET_PATH),
            split = args.split,
        )

    cleaned = clean_dataset(
        dataset,
        min_len=5,
        max_len=64,
        max_ratio=2.5,
    )

    test_dataset = TranslationDataset(
        cleaned,
        tokenizer,
        args.test_max_len,
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    print(f"Evaluating on {len(test_dataset)} sentences")

    bleu = evaluate(
        model,
        model_config,
        tokenizer,
        dataloader,
        device,
        args.test_max_len,
    )

    print("\n==============================")
    print(f"Average BLEU @ max_len={args.test_max_len}: {bleu:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    main()