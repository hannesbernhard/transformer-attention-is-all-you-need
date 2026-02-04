import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config.paths import GPT2_FROM_BPE, HYPERPARAMETERS, MODEL_CONFIG, BEST_MODELS
from src.modelling.model.transformer import TransformerConfig, TransformerModel
from src.utils.data_cleaning import clean_dataset
from src.utils.init_tokenizer import get_or_create_tokenizer
from src.dataset import TranslationDataset
import torch.nn.functional as F
from torch.optim import AdamW
import yaml
from tqdm import tqdm
import logging
import wandb
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
from pathlib import Path
import argparse

from src.utils.lr_scheduler import TransformerLRScheduler


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


def maybe_mount_drive(fetch_data_online: bool):
    if not fetch_data_online:
        return

    try:
        from google.colab import drive
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            drive.mount("/content/drive")
        else:
            print("Colab detected, but no IPython kernel — skipping drive.mount()")
    except Exception as e:
        print(f"Skipping drive.mount(): {e}")

class TrainerConfig:
    def __init__(self, **kwargs):
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 30)
        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        self.bleu_start_epoch = kwargs.get("bleu_start_epoch", 5)
        self.train_subset = kwargs.get("train_subset", 100000)
        self.val_subset = kwargs.get("val_subset", 10000)

        self.val_interval = kwargs.get("val_interval", 1)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")


class TransformerTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        config: TrainerConfig,
        model_config: dict,
        checkpoint_dir: Path,
    ):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.model_config = model_config
        self.checkpoint_dir = checkpoint_dir

        pin = self.config.device.type == "cuda"

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=pin,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            pin_memory=pin,
        )

        train_subset = len(train_dataset)
        total_steps = (train_subset // config.batch_size) * config.num_epochs
        self.warmup_steps = min(int(total_steps * 0.1), 35000)
        print(f"Total steps: {total_steps}, Warmup steps: {self.warmup_steps}")

        # ===== Correct AdamW initialization (no weight decay on bias & LayerNorm) =====
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "layernorm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        print(self.model_config)
        self.scheduler = TransformerLRScheduler(
            self.optimizer,
            self.model_config["d_model"],
            self.warmup_steps,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=0.1,
        )

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        wandb.init(project="transformer_experiment", config=config.__dict__)
        wandb.config.update(model_config)
        wandb.define_metric("Train", step_metric="epoch")
        wandb.define_metric("Validation", step_metric="epoch")
        wandb.define_metric("Learning Rate", step_metric="epoch")
        wandb.define_metric("BLEU", step_metric="epoch")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} Training",
            leave=False,
        )

        for batch in progress_bar:
            source_ids = batch["source_ids"].to(self.config.device)
            source_mask = batch["source_mask"].to(self.config.device)
            target_ids = batch["target_ids"].to(self.config.device)
            target_mask = batch["target_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            self.optimizer.zero_grad()
            output = self.model(
                source_ids, target_ids, source_mask, target_mask
            )
            logits = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            non_pad = labels != self.tokenizer.pad_token_id
            loss = F.cross_entropy(
                logits[non_pad],
                labels[non_pad],
                label_smoothing=0.1,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scheduler.step()
            self.optimizer.step()

            total_loss += loss.item()

        self.save_checkpoint(
            self.checkpoint_dir / "latest.pth",
            epoch,
        )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def calculate_bleu_generate(self, batch):
        source_ids = batch["source_ids"].to(self.config.device)
        labels = batch["labels"]

        preds, refs = [], []

        for i in range(source_ids.size(0)):
            generated = self.model.generate(
                source_ids[i : i + 1],
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=labels.size(1),
            )

            preds.append(
                self.tokenizer.decode(
                    generated[0], skip_special_tokens=True
                )
            )
            refs.append(
                self.tokenizer.decode(
                    labels[i], skip_special_tokens=True
                )
            )

        self.logger.info(
            f"Example:\nPredicted: {preds[0]}\nReference: {refs[0]}"
        )

        smooth = SmoothingFunction().method1
        scores = [
            sentence_bleu([r.split()], p.split(), smoothing_function=smooth)
            for p, r in zip(preds, refs)
        ]
        return sum(scores) / len(scores)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        bleu_scores = []

        progress_bar = tqdm(
            self.val_loader, desc=f"Epoch {epoch + 1} Validating"
        )
        for batch in progress_bar:
            source_ids = batch["source_ids"].to(self.config.device)
            source_mask = batch["source_mask"].to(self.config.device)
            target_ids = batch["target_ids"].to(self.config.device)
            target_mask = batch["target_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            output = self.model(
                source_ids, target_ids, source_mask, target_mask
            )

            if epoch >= self.config.bleu_start_epoch:
                bleu_scores.append(self.calculate_bleu_generate(batch))

            logits = output.view(-1, output.size(-1))
            labels = labels.view(-1)
            non_pad = labels != self.tokenizer.pad_token_id

            loss = F.cross_entropy(logits[non_pad], labels[non_pad])
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        if bleu_scores:
            wandb.log({"BLEU": sum(bleu_scores) / len(bleu_scores), "epoch": epoch + 1})
        wandb.log({"Validation": avg_loss, "epoch": epoch + 1})
        return avg_loss

    def train(self, start_epoch=0):
        best_val_loss = float("inf")

        for epoch in range(start_epoch, self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            train_loss = self.train_epoch(epoch)

            if (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate(epoch)
                lr = self.optimizer.param_groups[0]["lr"]

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": lr,
                    }
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pth",
                        epoch,
                    )

    def save_checkpoint(self, filename, epoch):
        filename.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "model_config": self.model_config,
            },
            filename,
        )


def main():
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)
    with open(HYPERPARAMETERS, "r") as f:
        hparams = yaml.safe_load(f)

    trainer_config = TrainerConfig(**hparams)

    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch_data_online", action="store_true")
    args = parser.parse_args()

    if args.fetch_data_online:
        train_ds = load_dataset(
            "wmt17", "de-en", split=f"train[{hparams['train_subset']}]"
        )
        val_ds = load_dataset(
            "wmt17",
            "de-en",
            split=f"validation[{hparams['val_subset']}]",
        )
    else:
        train_ds = load_dataset(
            str(DATASET_PATH),
            split=f"train[{hparams['train_subset']}]",
        )
        val_ds = load_dataset(
            str(DATASET_PATH),
            split=f"validation[{hparams['val_subset']}]",
        )

    max_length = model_config.get("max_len", 64)

    train_cleaned = clean_dataset(train_ds, 5, max_length, 2.5)
    val_cleaned = clean_dataset(val_ds, 5, max_length, 2.5)

    tokenizer = get_or_create_tokenizer()

    train_dataset = TranslationDataset(
        train_cleaned, tokenizer, max_length
    )
    val_dataset = TranslationDataset(val_cleaned, tokenizer, max_length)

    model = TransformerModel(
        TransformerConfig(**model_config)
    )

    maybe_mount_drive(args.fetch_data_online)

    checkpoint_dir = (
        Path("/content/drive/MyDrive/transformer_checkpoints")
        if args.fetch_data_online
        else BEST_MODELS
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerTrainer(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        trainer_config,
        model_config,
        checkpoint_dir,
    )

    resume = checkpoint_dir / "latest.pth"
    start_epoch = trainer.load_checkpoint(resume) if resume.exists() else 0
    trainer.train(start_epoch)


if __name__ == "__main__":
    main()
