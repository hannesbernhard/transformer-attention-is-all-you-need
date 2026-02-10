import torch
from datasets import load_dataset
import yaml

from src.modelling.model.transformer import TransformerModel, TransformerConfig
from src.utils.init_tokenizer import get_or_create_tokenizer

from config.paths import BEST_MODELS, MODEL_CONFIG

DEVICE = torch.device("cpu")



@torch.no_grad()
def teacher_forced_decode(model, tokenizer, src_text, tgt_text, max_len=64, device="cpu"):
    model.eval()

    src_ids = tokenizer.encode(
        src_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)

    tgt_ids = tokenizer.encode(
        tgt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)

    # shift right: decoder sees gold tokens
    logits = model(src_ids, tgt_ids[:, :-1])
    preds = logits.argmax(dim=-1)

    decoded = tokenizer.decode(preds[0], skip_special_tokens=True)
    return decoded


def load_model(checkpoint_path, use_rope):
    with open(MODEL_CONFIG, "r") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["use_rope"] = use_rope
    config = TransformerConfig(**model_cfg)

    model = TransformerModel(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

def run_teacher_forced(use_rope, checkpoint_path, example):
    tokenizer = get_or_create_tokenizer()
    model = load_model(checkpoint_path, use_rope)

    src_text = example["src"]
    tgt_text = example["tgt"]

    pred = teacher_forced_decode(
        model,
        tokenizer,
        src_text,
        tgt_text,
        max_len=64,
        device="cpu",
    )

    return tgt_text, pred


def main():
    with open("results/overfit_example.yaml") as f:
        example = yaml.safe_load(f)

    target, pred_sinus = run_teacher_forced(
        use_rope=False,
        checkpoint_path=BEST_MODELS/"sinus_overfit.pth",
        example=example,
    )

    _, pred_rope = run_teacher_forced(
        use_rope=True,
        checkpoint_path=BEST_MODELS/"rope_overfit.pth",
        example=example,
    )

    print("\nSOURCE (DE):")
    print(example["src"])

    print("\nTARGET (EN):")
    print(target)

    print("\nSINUS-PE (teacher-forced):")
    print(pred_sinus)

    print("\nRoPE (teacher-forced):")
    print(pred_rope)


if __name__ == "__main__":
    main()
