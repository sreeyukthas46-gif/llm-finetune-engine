"""
Trainer Module
Handles training loop, optimization, checkpointing, and metrics for fine-tuning LLMs.
Supports distributed training via Accelerate and provides utilities for monitoring training progress.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, cast
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD, Adam, AdamW as AdamWOptimizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers."""

    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_config: "OptimizerConfig",
    ) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.

        Args:
            model: The model to optimize.
            optimizer_config: Optimizer configuration.

        Returns:
            PyTorch optimizer instance.
        """
        optimizer_type = optimizer_config.optimizer_type.lower()
        params = model.parameters()
        
        if optimizer_type == "adamw":
            optimizer = AdamW(
                params,
                lr=optimizer_config.learning_rate,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
            )
        elif optimizer_type == "adam":
            optimizer = Adam(
                params,
                lr=optimizer_config.learning_rate,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
                weight_decay=optimizer_config.weight_decay,
            )
        elif optimizer_type == "sgd":
            optimizer = SGD(
                params,
                lr=optimizer_config.learning_rate,
                momentum=optimizer_config.momentum,
                weight_decay=optimizer_config.weight_decay,
                nesterov=optimizer_config.nesterov,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        logger.info(f"Created {optimizer_type.upper()} optimizer with LR={optimizer_config.learning_rate}")
        return optimizer


class LossFactory:
    """Factory for creating loss functions."""

    @staticmethod
    def create_loss_fn(loss_config: "LossConfig") -> nn.Module:
        """
        Create loss function based on configuration.

        Args:
            loss_config: Loss configuration.

        Returns:
            PyTorch loss function instance.
        """
        loss_type = loss_config.loss_type.lower()
        
        if loss_type == "crossentropy":
            loss_fn = CrossEntropyLoss(
                label_smoothing=loss_config.label_smoothing,
                ignore_index=loss_config.ignore_index,
                reduction=loss_config.reduction,
            )
        elif loss_type == "mse":
            loss_fn = nn.MSELoss(reduction=loss_config.reduction)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        logger.info(f"Created {loss_type.upper()} loss function")
        return loss_fn


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""

    @staticmethod
    def create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        total_steps: int,
        warmup_steps: int,
    ) -> Any:
        """
        Create learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule.
            scheduler_type: Type of scheduler ('linear', 'cosine', 'constant').
            total_steps: Total training steps.
            warmup_steps: Number of warmup steps.

        Returns:
            Learning rate scheduler instance.
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        logger.info(f"Created {scheduler_type.upper()} scheduler")
        return scheduler


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    
    optimizer_type: str = "adamw"  # "adamw", "adam", "sgd"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam-based optimizers
    eps: float = 1e-8
    momentum: float = 0.9  # For SGD
    nesterov: bool = True  # For SGD


@dataclass
class LossConfig:
    """Configuration for loss function."""
    
    loss_type: str = "crossentropy"  # "crossentropy", "mse", "custom"
    label_smoothing: float = 0.0
    ignore_index: int = -100
    reduction: str = "mean"  # "mean", "sum", "none"


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Training parameters
    num_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Optimizer and scheduler
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    lr_scheduler_type: str = "linear"  # "linear", "cosine", "constant"
    
    # Model and data
    model_name: str = "meta-llama/Llama-2-7b"
    output_dir: str = "./checkpoints"
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Distributed training
    use_accelerate: bool = True
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    
    # Logging
    logging_steps: int = 100
    log_model: bool = False
    
    # Seed for reproducibility
    seed: int = 42


class TrainerMetrics:
    """Track and manage training metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self.eval_accuracies: List[float] = []
        self.learning_rates: List[float] = []

    def log_train_loss(self, loss: float) -> None:
        """Log training loss."""
        self.train_losses.append(loss)

    def log_eval_metrics(self, eval_loss: float, accuracy: Optional[float] = None) -> None:
        """Log evaluation metrics."""
        self.eval_losses.append(eval_loss)
        if accuracy is not None:
            self.eval_accuracies.append(accuracy)

    def log_learning_rate(self, lr: float) -> None:
        """Log current learning rate."""
        self.learning_rates.append(lr)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert metrics to dictionary."""
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "eval_accuracies": self.eval_accuracies,
            "learning_rates": self.learning_rates,
        }

    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class Trainer:
    """Main trainer class for fine-tuning language models."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: str = "auto",
    ):
        """
        Initialize Trainer.

        Args:
            model: The model to train.
            tokenizer: HuggingFace tokenizer instance.
            train_dataloader: DataLoader for training data.
            eval_dataloader: DataLoader for evaluation data (optional).
            config: TrainingConfig instance with training hyperparameters.
            device: Device to train on ('cuda', 'cpu', or 'auto').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        self.device = self._set_device(device)
        
        # Initialize accelerator if available
        self.accelerator = None
        if self.config.use_accelerate and Accelerator is not None:
            self.accelerator = Accelerator(
                mixed_precision=self.config.mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
        
        # Setup loss function
        self.loss_fn = LossFactory.create_loss_fn(self.config.loss_config)
        
        # Setup optimizer and scheduler
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Any = None
        self._setup_optimizer_and_scheduler()
        
        # Metrics tracking
        self.metrics = TrainerMetrics()
        self.global_step = 0
        self.best_eval_loss = float("inf")
        
        logger.info("Trainer initialized successfully")

    @staticmethod
    def _set_device(device: str) -> str:
        """Set the appropriate device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _setup_optimizer_and_scheduler(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Create optimizer using factory
        self.optimizer = OptimizerFactory.create_optimizer(
            self.model,
            self.config.optimizer_config,
        )
        
        # Calculate total training steps
        total_steps = (
            len(self.train_dataloader)
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        
        # Warmup steps
        warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps > 0
            else int(total_steps * self.config.warmup_ratio)
        )
        
        # Create learning rate scheduler using factory
        self.lr_scheduler = SchedulerFactory.create_scheduler(
            self.optimizer,
            self.config.lr_scheduler_type,
            total_steps,
            warmup_steps,
        )
        
        logger.info(
            f"Optimizer and scheduler setup complete. "
            f"Total steps: {total_steps}, Warmup steps: {warmup_steps}"
        )

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        if self.accelerator is not None:
            # Accelerator handles device placement
            return batch
        
        # Manual device placement
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
        return prepared_batch

    def _compute_loss(self, model_outputs: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss from model outputs.

        Args:
            model_outputs: Model output object or tuple.
            batch: Input batch.

        Returns:
            Loss tensor.
        """
        # If model has built-in loss (like transformer models)
        if hasattr(model_outputs, "loss") and model_outputs.loss is not None:
            return model_outputs.loss
        
        # Otherwise, compute loss from logits and labels if available
        if "labels" in batch and hasattr(model_outputs, "logits"):
            logits = model_outputs.logits
            labels = batch["labels"]
            return self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        # Fallback: use model output directly as loss (not recommended)
        if isinstance(model_outputs, torch.Tensor):
            return model_outputs
        
        # If model outputs a tuple, try first element
        if isinstance(model_outputs, tuple) and len(model_outputs) > 0:
            first_output = model_outputs[0]
            if isinstance(first_output, torch.Tensor) and first_output.dim() == 0:
                return first_output
        
        raise ValueError(
            "Cannot compute loss from model outputs. "
            "Ensure model returns loss or (logits, labels) in batch."
        )

    def _update_metrics_batch(self, loss: float, batch_size: int) -> None:
        """Update metrics for a batch."""
        self.metrics.log_train_loss(loss)

    def train(self) -> Tuple[Dict[str, Any], TrainerMetrics]:
        """
        Train the model.

        Returns:
            Tuple of training results and metrics.
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Setup output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Prepare for distributed training if using Accelerate
        if self.accelerator is not None:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
                )
            )
        else:
            self.model = self.model.to(self.device)
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = self._train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}")
            
            # Evaluation
            if self.eval_dataloader is not None:
                eval_loss = self._evaluate()
                logger.info(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint(
                        checkpoint_dir=os.path.join(
                            self.config.output_dir, "best_model"
                        ),
                        is_best=True,
                    )
        
        # Final checkpoint
        self._save_checkpoint(
            checkpoint_dir=os.path.join(self.config.output_dir, "final_model")
        )
        
        # Save metrics
        metrics_path = os.path.join(self.config.output_dir, "metrics.json")
        self.metrics.save(metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")
        
        training_results = {
            "num_epochs": self.config.num_epochs,
            "total_steps": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "final_train_loss": epoch_loss if epoch is not None else None,
        }
        
        return training_results, self.metrics

    def _train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average loss for the epoch.
        """
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {epoch + 1}",
            disable=self.accelerator is not None and not self.accelerator.is_main_process,
        )
        
        for step, batch in enumerate(progress_bar):
            batch = self._prepare_batch(batch)
            
            # Forward pass
            if self.accelerator is not None:
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs, batch)
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    
                    if self.optimizer is not None:
                        self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
            else:
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    if self.optimizer is not None:
                        self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.optimizer is not None:
                        self.optimizer.zero_grad()
                    self.global_step += 1
            
            # Logging
            epoch_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            num_batches += 1
            
            if self.global_step % self.config.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else 0.0
                self.metrics.log_learning_rate(current_lr)
                avg_loss = epoch_loss / num_batches
                self.metrics.log_train_loss(avg_loss)
                logger.info(
                    f"Step {self.global_step}: Loss = {avg_loss:.4f}, LR = {current_lr:.2e}"
                )
            
            progress_bar.set_postfix({"loss": epoch_loss / num_batches})
        
        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def _evaluate(self) -> float:
        """
        Evaluate the model.

        Returns:
            Average evaluation loss.
        """
        logger.info("Running evaluation...")
        self.model.eval()
        
        eval_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.eval_dataloader,
                desc="Evaluation",
                disable=self.accelerator is not None and not self.accelerator.is_main_process,
            )
            
            for batch in progress_bar:
                batch = self._prepare_batch(batch)
                
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
                
                if self.accelerator is not None:
                    gathered_loss = self.accelerator.gather(loss.unsqueeze(0))
                    if isinstance(gathered_loss, torch.Tensor):
                        loss = gathered_loss.mean().item()
                    else:
                        loss = float(loss.item())
                else:
                    loss = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                
                eval_loss += loss
                num_batches += 1
                progress_bar.set_postfix({"loss": eval_loss / num_batches})
        
        avg_eval_loss = eval_loss / num_batches if num_batches > 0 else 0.0
        self.metrics.log_eval_metrics(avg_eval_loss)
        
        self.model.train()
        return avg_eval_loss

    def _save_checkpoint(
        self, checkpoint_dir: str, is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint.
            is_best: Whether this is the best model so far.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if self.accelerator is not None:
            self.accelerator.save_model(self.model, checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save optimizer and scheduler state
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(checkpoint_dir, "optimizer.pt"),
            )
        if self.lr_scheduler is not None:
            torch.save(
                self.lr_scheduler.state_dict(),
                os.path.join(checkpoint_dir, "scheduler.pt"),
            )
        
        # Save training config
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "w") as f:
            config_dict = {
                k: v
                for k, v in self.config.__dict__.items()
                if not callable(v)
            }
            json.dump(config_dict, f, indent=2)
        
        checkpoint_type = "best" if is_best else "checkpoint"
        logger.info(f"{checkpoint_type.capitalize()} saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoint files.
        """
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist")
            return
        
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path) and self.optimizer is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path))
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Update model",
    ) -> None:
        """
        Push model to HuggingFace Hub.

        Args:
            repo_id: HuggingFace Hub repository ID.
            private: Whether the repository should be private.
            commit_message: Commit message for the push.
        """
        try:
            # Get the underlying model if using accelerator
            if self.accelerator is not None:
                model_to_push = self.accelerator.unwrap_model(self.model)
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    push_to_hub_fn = getattr(model_to_push, 'push_to_hub', None)
                    if push_to_hub_fn is not None and callable(push_to_hub_fn):
                        push_to_hub_fn(
                            repo_id,
                            private=private,
                            commit_message=commit_message,
                        )
            else:
                push_to_hub_fn = getattr(self.model, 'push_to_hub', None)
                if push_to_hub_fn is not None and callable(push_to_hub_fn):
                    push_to_hub_fn(
                        repo_id,
                        private=private,
                        commit_message=commit_message,
                    )
            logger.info(f"Model pushed to {repo_id}")
        except Exception as e:
            logger.error(f"Failed to push model to hub: {e}")
