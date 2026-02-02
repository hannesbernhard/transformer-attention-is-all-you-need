from datetime import datetime
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config.paths import GPT2_FROM_BPE, HYPERPARAMETERS, MODEL_CONFIG, BEST_MODELS
from src.modelling.model.transformer import TransformerConfig, TransformerModel
from src.utils.data_cleaning import clean_dataset
from src.utils.init_tokenizer import get_or_create_tokenizer
from transformers import GPT2Tokenizer
from src.dataset import TranslationDataset
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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
DATASET_PATH = os.path.join(project_root, "hf_cache", "root", ".cache", "huggingface", "datasets", "wmt17", "de-en", "0.0.0", "54d3aacfb5429020b9b85b170a677e4bc92f2449")


class TrainerConfig:
    def __init__(self, **kwargs):
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.bleu_start_epoch = kwargs.get("bleu_start_epoch", 5)
        self.train_subset = kwargs.get("train_subset", 100000)
        self.val_subset = kwargs.get("val_subset", 10000)

        self.fetch_data_online = kwargs.get("fetch_data_online", False)

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
    ):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.model_config = model_config

        pin = self.config.device.type == "cuda"

        # Initialize dataloaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=pin
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=pin
        )

        train_subset = len(train_dataset)
        total_steps = (train_subset // config.batch_size) * config.num_epochs
        self.warmup_steps = min(int(total_steps * 0.1), 35000)
        print(f"Total steps: {total_steps}, Warmup steps: {self.warmup_steps}")

        # Initialize optimizer and scheduler
        # parameters from the paper
        self.optimizer = AdamW(
            model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9
        )
        print(self.model_config)
        self.scheduler = TransformerLRScheduler(
            self.optimizer, self.model_config["d_model"], self.warmup_steps
        )

        # Initialize loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
        )

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize WandB
        wandb.init(project="transformer_experiment", config=config.__dict__)
        wandb.config.update(model_config)
        wandb.define_metric("Train", step_metric="epoch")
        wandb.define_metric("Validation", step_metric="epoch")
        wandb.define_metric("Learning Rate", step_metric="epoch")
        wandb.define_metric("BLEU", step_metric="epoch")

    def train_epoch(self, epoch):
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} Training",
            leave=False,
        )

        for batch in progress_bar:
            # Move batch to device
            source_ids = batch["source_ids"].to(self.config.device)
            source_mask = batch["source_mask"].to(self.config.device)
            target_ids = batch["target_ids"].to(self.config.device)
            target_mask = batch["target_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(source_ids, target_ids, source_mask, target_mask)
            logits = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            # mask PAD tokens
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

            # Update metrics
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def calculate_bleu_generate(self, batch):
        source_ids = batch["source_ids"].to(self.config.device)
        labels = batch["labels"]

        preds = []
        refs = []

        for i in range(source_ids.size(0)):
            generated = self.model.generate(
                source_ids[i:i+1],
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=labels.size(1),
            )

            pred_text = self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
            ref_text = self.tokenizer.decode(
                labels[i], skip_special_tokens=True
            )

            preds.append(pred_text)
            refs.append(ref_text)

        # Log one example during validation
        self.logger.info(
            f"Example from validation set: \nPredicted: {preds[0]} \nReference: {refs[0]}"
        )

        smooth = SmoothingFunction().method1
        scores = [
            sentence_bleu([r.split()], p.split(), smoothing_function=smooth)
            for p, r in zip(preds, refs)
        ]

        return sum(scores) / len(scores)

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        all_bleu_scores = []

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} Validating")
        for batch in progress_bar:
            # Move batch to device
            source_ids = batch["source_ids"].to(self.config.device)
            source_mask = batch["source_mask"].to(self.config.device)
            target_ids = batch["target_ids"].to(self.config.device)
            target_mask = batch["target_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # Forward pass
            output = self.model(source_ids, target_ids, source_mask, target_mask)
            if epoch >= self.config.bleu_start_epoch:
                bleu_score = self.calculate_bleu_generate(batch)
                all_bleu_scores.append(bleu_score)

            logits = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            # Calculate loss
            non_pad = labels != self.tokenizer.pad_token_id
            loss = F.cross_entropy(
                logits[non_pad],
                labels[non_pad],
            )
            total_loss += loss.item()

            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.val_loader)
        if len(all_bleu_scores) > 0:
            avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
            wandb.log({"BLEU": avg_bleu, "epoch": epoch + 1})
        wandb.log({"Validation": avg_loss, "epoch": epoch + 1})
        return avg_loss

    def train(self):
        """Main training loop."""
        best_val_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            # log current learning rate
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            train_loss = self.train_epoch(epoch)

            # Validation phase
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate(epoch)
                current_lr = self.optimizer.param_groups[0]["lr"]

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": current_lr,
                    }
                )

                self.logger.info(
                    f"Epoch {epoch + 1} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"lr={current_lr:.2e}"
                )

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(BEST_MODELS / "best_model.pth")
                    self.logger.info("New best model saved!")

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": self.model_config,
        }
        torch.save(checkpoint, filename)


def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch_data_online", type=str, required=False)
    args = parser.parse_args()

    return args.fetch_data_online


def main():
    # Load configurations
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)
    with open(HYPERPARAMETERS, "r") as f:
        hparams = yaml.safe_load(f)

    print("Model Config:")
    print(model_config)

    print("Hyperparameters:")
    print(hparams)

    # Create trainer config
    trainer_config = TrainerConfig(**hparams)

    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fetch_data_online",
        action="store_true",
        help="Fetch dataset from HuggingFace instead of local files"
    )
    args = parser.parse_args()

    if args.fetch_data_online:
        train_ds = load_dataset("wmt17", "de-en", split=f"train[{hparams['train_subset']}]")
        val_ds = load_dataset("wmt17", "de-en", split=f"validation[{hparams["val_subset"]}]")
 
    else:
        # Load and prepare datasets
        train_ds = load_dataset(str(DATASET_PATH), split=f"train[{hparams['train_subset']}]")
        val_ds   = load_dataset(str(DATASET_PATH), split=f"validation[{hparams["val_subset"]}]")

    max_length = model_config.get("max_len", 64)
    train_cleaned = clean_dataset(
        train_ds,
        min_len=5,
        max_len=max_length,
        max_ratio=2.5,
    )

    val_cleaned = clean_dataset(
        val_ds,
        min_len=5,
        max_len=max_length,
        max_ratio=2.5,
    )
    print("train dataset size: ", len(train_cleaned))
    print("val dataset size: ", len(val_cleaned))

    # Initialize tokenizer
    tokenizer = get_or_create_tokenizer()

    # Create datasets
    train_dataset = TranslationDataset(train_cleaned, tokenizer, max_length=max_length)
    val_dataset = TranslationDataset(val_cleaned, tokenizer, max_length=max_length)

    # Initialize model
    transformer_config = TransformerConfig(**model_config)
    model = TransformerModel(config=transformer_config)

    # Create trainer and start training
    trainer = TransformerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        model_config=model_config,
    )

    trainer.train()


if __name__ == "__main__":
    main()
