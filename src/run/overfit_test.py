import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from src.modelling.model.transformer import TransformerModel, TransformerConfig
from src.dataset import TranslationDataset
from src.utils.data_cleaning import clean_dataset
from src.utils.init_tokenizer import get_or_create_tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ===== Tiny dataset =====
    raw = load_dataset("wmt17", "de-en", split="train[:20]")
    cleaned = clean_dataset(raw, min_len=5, max_len=64, max_ratio=2.5)

    tokenizer = get_or_create_tokenizer()
    dataset = TranslationDataset(cleaned, tokenizer, max_length=64)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # ===== Tiny model =====
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.0,
        max_len=64,
        use_rope=True,
    )

    model = TransformerModel(config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    for epoch in range(200):
        total_loss = 0.0

        for batch in loader:
            src = batch["source_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            logits = model(src, tgt)
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

            loss = F.cross_entropy(
                logits,
                labels,
                ignore_index=tokenizer.pad_token_id,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f}")

        # Early success condition
        if avg_loss < 0.05:
            print("Successfully overfitted!")
            break

    # ===== Qualitative check =====
    model.eval()
    with torch.no_grad():
        example = dataset[0]
        src = example["source_ids"].unsqueeze(0).to(device)

        generated = model.generate(
            src,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=64,
        )

        print("\nSOURCE:")
        print(tokenizer.decode(example["source_ids"], skip_special_tokens=True))
        print("\nTARGET:")
        print(tokenizer.decode(example["labels"], skip_special_tokens=True))
        print("\nPREDICTED:")
        print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
