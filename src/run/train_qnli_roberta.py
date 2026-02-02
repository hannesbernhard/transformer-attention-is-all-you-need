# QNLI Entailment Training with RoBERTa
# GPU + Mixed Precision Support

import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.metrics import f1_score
import tqdm


# Configuration
MODEL_NAME_OR_PATH = "roberta-base"
MAX_INPUT_LENGTH = 256
BATCH_SIZE = 16
TRAINING_EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION_TRAINING = torch.cuda.is_available()

print(f"Using device: {DEVICE}")
print(f"Mixed precision: {MIXED_PRECISION_TRAINING}")


# Model & Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH,
    num_labels=2
)
model.to(DEVICE)


# Dataset
qnli_dataset = load_dataset("glue", "qnli")

def convert_example_to_features(example):
    features = tokenizer(
        example["question"],
        example["sentence"],
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation="longest_first",
    )
    features["labels"] = example["label"]
    return features

def collate(batch):
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch]),
        "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }

train_dataset = qnli_dataset["train"].map(convert_example_to_features)
validation_dataset = qnli_dataset["validation"].map(convert_example_to_features)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate,
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate,
)


# Optimizer & Scheduler
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": WEIGHT_DECAY,
        "lr": LEARNING_RATE,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": LEARNING_RATE,
    },
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

num_training_steps = len(train_dataloader) * TRAINING_EPOCHS
num_warmup_steps = int(WARMUP_PROPORTION * num_training_steps)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION_TRAINING)


# Training & Evaluation Functions
def training_step(batch):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    with torch.amp.autocast(enabled=MIXED_PRECISION_TRAINING):
        outputs = model(**batch)
        loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.detach().cpu().item()

def evaluate(dataloader):
    model.eval()
    predictions, labels = [], []

    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(**batch).logits

        preds = logits.argmax(dim=-1)
        predictions.append(preds.cpu())
        labels.append(batch["labels"].cpu())

    model.train()

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    return f1_score(labels, predictions)


# Training Loop
model.train()
optimizer.zero_grad()

for epoch in range(TRAINING_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{TRAINING_EPOCHS}")
    iterator = tqdm.tqdm(train_dataloader, desc="Training")

    for batch in iterator:
        loss = training_step(batch)
        iterator.set_postfix({"loss": f"{loss:.4f}"})

    f1 = evaluate(validation_dataloader)
    print(f"Validation F1 score: {f1:.4f}")

print("Training complete.")
