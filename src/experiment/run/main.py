import torch
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

from config.paths import (
    BEST_MODELS,
    EXPERIMENT_DATA,
    EXPERIMENT_STORAGE,
    GPT2_FROM_BPE,
    MODEL_CONFIG,
)
from src.experiment.models.SoftPrompt import setup_soft_prompting_model
from src.experiment.data.synthetic_data import create_dataloader
from src.experiment.models.LoRa import setup_lora_model
from src.modelling.model.transformer import TransformerConfig, TransformerModel
from src.utils.lr_scheduler import TransformerLRScheduler
from transformers import GPT2Tokenizer


class ExperimentTrainer:
    def __init__(self, base_model, tokenizer, config, rank):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps:0")
        print(self.device)
        self.base_model = base_model.to(self.device)
        self.tokenizer = tokenizer

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Setup paths
        self.results_path = Path("results")
        self.lora_path = self.results_path / "lora"
        self.soft_prompt_path = self.results_path / "soft_prompt"

        # Create directories
        self.lora_path.mkdir(parents=True, exist_ok=True)
        self.soft_prompt_path.mkdir(parents=True, exist_ok=True)

        wandb.init(project="transformer_experiment_lora", name=f"LoRA_Rank_{rank}")
        wandb.define_metric("Train", step_metric="epoch")
        wandb.define_metric("Validation", step_metric="epoch")
        wandb.define_metric("Learning Rate", step_metric="epoch")

    def train_lora(self, train_loader, val_loader, rank=8):
        """Train LoRA model"""
        self.logger.info("Starting LoRA training...")

        # Initialize LoRA model
        lora_model = setup_lora_model(self.base_model, rank)
        lora_model = lora_model.to(self.device)

        optimizer = torch.optim.AdamW(
            [p for p in lora_model.parameters() if p.requires_grad], lr=0.5
        )

        scheduler = TransformerLRScheduler(optimizer, 128, warmup_steps=1000)

        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id
        )  # 0 is pad_token_id

        # Training loop
        best_val_loss = float("inf")

        # Initialize WandB

        for epoch in range(self.config["num_epochs"]):
            # Training phase
            lora_model.train()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"Learning Rate": current_lr, "epoch": epoch})
            train_loss = self._train_epoch(
                epoch, lora_model, train_loader, optimizer, scheduler, criterion
            )
            self.logger.info(f"Training loss: {train_loss:.4f}")

            # Validation phase
            val_loss = self._validate(epoch, lora_model, val_loader, criterion)

            # Logging

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.logger.info(f"Saving best model for rank: {rank}...")
                torch.save(
                    lora_model,
                    EXPERIMENT_STORAGE / "LoRa" / f"r_{rank}" / "best_model.pth",
                )

        wandb.finish()
        return lora_model

    def train_soft_prompting(
        self, train_loader, val_loader, prompt_length, embedding_dim
    ):
        """Train Soft-Prompting model."""
        self.logger.info("Starting Soft-Prompting training...")

        soft_prompt_model = setup_soft_prompting_model(
            self.base_model, prompt_length, embedding_dim
        )
        soft_prompt_model = soft_prompt_model.to(self.device)

        # Only optimize the soft prompts
        optimizer = torch.optim.AdamW(
            [p for p in soft_prompt_model.soft_prompting_layer.parameters()], lr=0.5
        )

        scheduler = TransformerLRScheduler(optimizer, 128, warmup_steps=1000)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # Training loop
        best_val_loss = float("inf")

        for epoch in range(self.config["num_epochs"]):
            # Training phase
            soft_prompt_model.train()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"Learning Rate": current_lr, "epoch": epoch})
            train_loss = self._train_epoch(
                epoch, soft_prompt_model, train_loader, optimizer, scheduler, criterion
            )
            self.logger.info(f"Training loss: {train_loss:.4f}")

            # Validation phase
            val_loss = self._validate(epoch, soft_prompt_model, val_loader, criterion)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    soft_prompt_model,
                    EXPERIMENT_STORAGE
                    / "soft_prompt"
                    / "multi_task"
                    / "best_model.pth",
                )

        wandb.finish()
        return soft_prompt_model

    def _train_epoch(self, epoch, model, loader, optimizer, scheduler, criterion):
        """Run one epoch of training."""
        total_loss = 0

        for batch in tqdm(loader, desc="Training"):
            source_ids = batch["input_ids"].to(self.device)
            source_mask = batch["input_mask"].to(self.device)
            target_ids = batch["target_input"].to(self.device)
            target_mask = batch["target_mask"].to(self.device)
            labels = batch["target_label"].to(self.device)

            optimizer.zero_grad()
            output = model(source_ids, target_ids, source_mask, target_mask)
            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        wandb.log({"Train": avg_loss, "epoch": epoch + 1})
        return avg_loss

    def calculate_bleu_from_logits(self, logits, labels, _inputs):
        """
        Calculate the BLEU score for a batch of logits and labels.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, seq_len, vocab_size).
            labels (torch.Tensor): Ground-truth labels of shape (batch_size, seq_len).

        Returns:
            float: Average BLEU score for the batch.
        """

        # Convert logits to predicted token IDs
        predicted_ids = logits.argmax(dim=-1)  # Shape: (batch_size, seq_len)

        preds = []
        for pred_seq in predicted_ids:
            # Find the index of [EOS] in pred_seq
            eos_indices = torch.nonzero(
                pred_seq == self.tokenizer.eos_token_id, as_tuple=True
            )[0]
            if len(eos_indices) > 0:
                eos_index = eos_indices[0].item()  # First occurrence of [EOS]
                pred_seq = pred_seq[: eos_index + 1]  # Include [EOS] itself

            # Decode the sequence
            preds.append(
                self.tokenizer.decode(pred_seq.tolist(), skip_special_tokens=True)
            )

        refs = [
            self.tokenizer.decode(ref_seq.tolist(), skip_special_tokens=True)
            for ref_seq in labels
        ]

        # Log one example during validation
        self.logger.info(
            f"Example from validation set: \ninput: {_inputs[0]} \nPredicted: {preds[0]} \nReference: {refs[0]}"
        )

        return 1

    def _validate(self, epoch, model, loader, criterion):
        """Run validation."""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validating"):
                _input = batch["input"]
                source_ids = batch["input_ids"].to(self.device)
                source_mask = batch["input_mask"].to(self.device)
                target_ids = batch["target_input"].to(self.device)
                target_mask = batch["target_mask"].to(self.device)
                labels = batch["target_label"].to(self.device)

                output = model(source_ids, target_ids, source_mask, target_mask)

                bleu_score = self.calculate_bleu_from_logits(output, labels, _input)

                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        wandb.log({"Validation": avg_loss, "epoch": epoch + 1})
        return avg_loss


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
    # Load your base transformer model
    with open(MODEL_CONFIG, "r") as f:
        model_config = yaml.safe_load(f)

    base_model = load_model(BEST_MODELS / "best_model.pth", model_config)

    config = {"batch_size": 128, "num_epochs": 40, "learning_rate": 0.5, "rank": 32}

    # File paths
    train_file = EXPERIMENT_DATA / "multi_task" / "train_data.json"
    val_file = EXPERIMENT_DATA / "multi_task" / "val_data.json"

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(str(GPT2_FROM_BPE / "64"))
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",  # 0
            "bos_token": "[BOS]",  # 1
            "eos_token": "[EOS]",  # 2
            "unk_token": "[UNK]",  # 3
        }
    )

    train_loader = create_dataloader(
        train_file, config["batch_size"], tokenizer, shuffle=True
    )
    val_loader = create_dataloader(
        val_file, config["batch_size"], tokenizer, shuffle=False
    )

    # Initialize trainer

    # Train both models
    rank = config["rank"]
    trainer = ExperimentTrainer(base_model, tokenizer, config, rank)
    lora_model = trainer.train_lora(train_loader, val_loader, rank=rank)


if __name__ == "__main__":
    main()
