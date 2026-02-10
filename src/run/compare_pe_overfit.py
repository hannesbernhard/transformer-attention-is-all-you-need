import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path

from config.paths import BEST_MODELS
from src.modelling.model.transformer import TransformerModel, TransformerConfig
from src.dataset import TranslationDataset
from src.utils.data_cleaning import clean_dataset
from src.utils.init_tokenizer import get_or_create_tokenizer


Path("results").mkdir(exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_experiment(use_rope: bool, save_name: str):
    set_seed(42)

    device = torch.device("cpu")
    print(f"\n=== Running with use_rope={use_rope} ===")

    raw = load_dataset("wmt17", "de-en", split="train[:20]")
    cleaned = clean_dataset(raw, 5, 64, 2.5)

    tokenizer = get_or_create_tokenizer()
    dataset = TranslationDataset(cleaned, tokenizer, max_length=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        max_len=64,
        use_rope=use_rope,
    )

    model = TransformerModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []

    example = cleaned[0]   # oder der Satz, den du nutzt

    with open("results/overfit_example.yaml", "w") as f:
        yaml.dump(example, f)

    model.train()
    for epoch in range(80):
        total_loss = 0.0
        for batch in loader:
            src = batch["source_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(src, tgt)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        losses.append(avg)
        print(f"Epoch {epoch:02d} | Loss {avg:.4f}")

    torch.save(
        {"model_state_dict": model.state_dict()},
        BEST_MODELS / save_name,
    )

    return losses


def main():
    losses_sinus = run_experiment(use_rope=False, save_name="sinus_overfit.pth")
    losses_rope = run_experiment(use_rope= True ,save_name="rope_overfit.pth")

    print("\n=== Summary ===")
    print(f"Final loss (Sinus-PE): {losses_sinus[-1]:.4f}")
    print(f"Final loss (RoPE):     {losses_rope[-1]:.4f}")


    plt.plot(losses_sinus, label="Sinus-PE")
    plt.plot(losses_rope, label="RoPE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Overfit Comparison: Sinusoidal PE vs RoPE")
    plt.show()


if __name__ == "__main__":
    main()

