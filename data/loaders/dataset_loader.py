"""
Dataset Loader Module
Handles loading, processing, and preparing datasets for fine-tuning LLMs.
Supports various data formats and provides utilities for tokenization and batching.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""

    data_path: str
    max_length: int = 512
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 8
    shuffle: bool = True
    seed: int = 42


class InstructionDataset(Dataset):
    """PyTorch Dataset for instruction-based fine-tuning (instruction/input/output format)."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        instruction_template: str = "[INST] {instruction}\n{input} [/INST] {output}",
    ):
        """
        Initialize InstructionDataset.

        Args:
            data: List of dictionaries with 'instruction', 'input', and 'output' keys.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum token length for sequences.
            instruction_template: Template for formatting instruction data.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        sample = self.data[idx]

        # Format the instruction
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        # Create the full prompt
        prompt = self.instruction_template.format(
            instruction=instruction, input=input_text, output=output_text
        )

        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Labels are the same as input_ids for causal language modeling
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SimpleDataset(Dataset):
    """PyTorch Dataset for simple text-only fine-tuning."""

    def __init__(
        self,
        data: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize SimpleDataset.

        Args:
            data: List of text samples.
            tokenizer: HuggingFace tokenizer instance.
            max_length: Maximum token length for sequences.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'labels'.
        """
        text = self.data[idx]

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DatasetLoader:
    """Loader for managing and preparing datasets for fine-tuning."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Optional[DatasetConfig] = None,
    ):
        """
        Initialize DatasetLoader.

        Args:
            tokenizer: HuggingFace tokenizer instance.
            config: Dataset configuration object.
        """
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig(data_path="data/raw/dataset.json")
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        logger.info(f"DatasetLoader initialized with config: {self.config}")

    def load_json_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load data from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            List of data samples.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON format is invalid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of samples")

            logger.info(f"Loaded {len(data)} samples from {file_path}")
            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")

    def load_text_data(self, file_path: Union[str, Path]) -> List[str]:
        """
        Load data from a text file (one sample per line).

        Args:
            file_path: Path to the text file.

        Returns:
            List of text samples.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def create_instruction_dataset(
        self,
        data: List[Dict[str, str]],
        max_length: Optional[int] = None,
        instruction_template: Optional[str] = None,
    ) -> InstructionDataset:
        """
        Create an instruction-based dataset.

        Args:
            data: List of instruction samples.
            max_length: Maximum sequence length.
            instruction_template: Custom template for formatting.

        Returns:
            InstructionDataset instance.
        """
        max_length = max_length or self.config.max_length
        instruction_template = (
            instruction_template or "[INST] {instruction}\n{input} [/INST] {output}"
        )

        return InstructionDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=max_length,
            instruction_template=instruction_template,
        )

    def create_simple_dataset(
        self,
        data: List[str],
        max_length: Optional[int] = None,
    ) -> SimpleDataset:
        """
        Create a simple text-only dataset.

        Args:
            data: List of text samples.
            max_length: Maximum sequence length.

        Returns:
            SimpleDataset instance.
        """
        max_length = max_length or self.config.max_length
        return SimpleDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=max_length,
        )

    def split_dataset(
        self,
        dataset: Union[InstructionDataset, SimpleDataset],
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> Tuple[
        Union[Subset, InstructionDataset, SimpleDataset],
        Union[Subset, InstructionDataset, SimpleDataset],
        Union[Subset, InstructionDataset, SimpleDataset],
    ]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            dataset: Dataset to split.
            train_ratio: Proportion for training (default from config).
            val_ratio: Proportion for validation (default from config).
            test_ratio: Proportion for testing (default from config).

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            ValueError: If ratios don't sum to approximately 1.0.
        """
        train_ratio = train_ratio or self.config.train_split
        val_ratio = val_ratio or self.config.val_split
        test_ratio = test_ratio or self.config.test_split

        # Normalize ratios
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Ratios must sum to ~1.0, got {total}. "
                f"Provided: train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )

        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

        dataset_size = len(dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        logger.info(
            f"Dataset split: train={train_size}, val={val_size}, test={test_size}"
        )

        return train_dataset, val_dataset, test_dataset

    def create_dataloader(
        self,
        dataset: Union[InstructionDataset, SimpleDataset, Subset],
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader from a dataset.

        Args:
            dataset: Dataset to load.
            batch_size: Batch size (default from config).
            shuffle: Whether to shuffle data (default from config).
            num_workers: Number of worker processes.

        Returns:
            DataLoader instance.
        """
        batch_size = batch_size or self.config.batch_size
        shuffle = shuffle if shuffle is not None else self.config.shuffle

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def load_and_prepare(
        self,
        data_path: Optional[str] = None,
        data_type: str = "instruction",
        split: bool = True,
        create_loaders: bool = True,
    ) -> Dict[str, Any]:
        """
        Load and prepare data end-to-end.

        Args:
            data_path: Path to data file (uses config path if not provided).
            data_type: Type of data ('instruction' or 'text').
            split: Whether to split into train/val/test.
            create_loaders: Whether to create DataLoaders.

        Returns:
            Dictionary with datasets and dataloaders.

        Raises:
            ValueError: If data_type is invalid.
        """
        data_path = data_path or self.config.data_path

        if data_type not in ["instruction", "text"]:
            raise ValueError(
                f"data_type must be 'instruction' or 'text', got '{data_type}'"
            )

        # Load raw data
        if data_type == "instruction":
            raw_data = self.load_json_data(data_path)
            dataset = self.create_instruction_dataset(raw_data)
        else:
            raw_data = self.load_text_data(data_path)
            dataset = self.create_simple_dataset(raw_data)

        result = {
            "full_dataset": dataset,
            "train_dataset": None,
            "val_dataset": None,
            "test_dataset": None,
            "train_loader": None,
            "val_loader": None,
            "test_loader": None,
        }

        # Split if requested
        if split:
            train_dataset, val_dataset, test_dataset = self.split_dataset(dataset)
            result["train_dataset"] = train_dataset
            result["val_dataset"] = val_dataset
            result["test_dataset"] = test_dataset

            if create_loaders:
                result["train_loader"] = self.create_dataloader(
                    train_dataset, shuffle=True
                )
                result["val_loader"] = self.create_dataloader(
                    val_dataset, shuffle=False
                )
                result["test_loader"] = self.create_dataloader(
                    test_dataset, shuffle=False
                )
        else:
            if create_loaders:
                result["train_loader"] = self.create_dataloader(
                    dataset, shuffle=self.config.shuffle
                )

        logger.info("Data loading and preparation completed successfully")
        return result

    def get_dataset_statistics(self, dataset: Union[InstructionDataset, SimpleDataset, Subset]) -> Dict[str, Any]:
        """
        Get statistics about a dataset.

        Args:
            dataset: Dataset to analyze.

        Returns:
            Dictionary with dataset statistics.
        """
        sizes = []
        dataset_len = len(dataset)  # type: ignore
        for i in range(dataset_len):
            sample = dataset[i]  # type: ignore
            size = sample["input_ids"].shape[0]  # type: ignore
            sizes.append(size)

        sizes = np.array(sizes)

        stats = {
            "num_samples": dataset_len,
            "avg_sequence_length": float(np.mean(sizes)),
            "min_sequence_length": int(np.min(sizes)),
            "max_sequence_length": int(np.max(sizes)),
            "std_sequence_length": float(np.std(sizes)),
        }

        logger.info(f"Dataset statistics: {stats}")
        return stats
