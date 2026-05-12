"""
Model Loader Module
Handles loading pretrained models from HuggingFace with support for
quantization, PEFT methods, and various configurations.
"""

import logging
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and configure pretrained language models."""

    def __init__(self, device: str = "auto"):
        """
        Initialize ModelLoader.

        Args:
            device: Device to load model on ('cuda', 'cpu', or 'auto')
        """
        self.device = self._set_device(device)
        logger.info(f"ModelLoader initialized with device: {self.device}")

    @staticmethod
    def _set_device(device: str) -> str:
        """
        Set the appropriate device for model loading.

        Args:
            device: Requested device ('cuda', 'cpu', or 'auto')

        Returns:
            str: The device to use
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_pretrained_model(
        self,
        model_name: str,
        dtype: str = "fp32",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_lora: bool = False,
        lora_config: Optional[Dict[str, Any]] = None,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ) -> tuple:
        """
        Load a pretrained model from HuggingFace.

        Args:
            model_name: HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b')
            dtype: Data type for model weights ('fp32', 'fp16', 'bfloat16')
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
            use_lora: Whether to apply LoRA fine-tuning
            lora_config: Custom LoRA configuration
            trust_remote_code: Allow loading custom modeling code from HuggingFace
            cache_dir: Directory to cache downloaded models

        Returns:
            tuple: (model, tokenizer)

        Raises:
            ValueError: If invalid configuration is provided
            RuntimeError: If model loading fails
        """
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization")

        logger.info(f"Loading pretrained model: {model_name}")

        try:
            # Configure quantization if specified
            quantization_config = None
            device_map = "auto" if self.device == "cuda" else None

            if load_in_8bit or load_in_4bit:
                quantization_config = self._create_quantization_config(
                    load_in_8bit, load_in_4bit
                )

            # Load tokenizer
            tokenizer = self._load_tokenizer(model_name, cache_dir)

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._get_torch_dtype(dtype),
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            )

            logger.info(f"Model loaded successfully: {model_name}")

            # Apply LoRA if specified
            if use_lora:
                model = self._apply_lora(model, lora_config)
                logger.info("LoRA fine-tuning adapter applied")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}") from e

    def _load_tokenizer(
        self, model_name: str, cache_dir: Optional[str] = None
    ):
        """
        Load tokenizer for the model.

        Args:
            model_name: HuggingFace model ID
            cache_dir: Directory to cache downloaded tokenizers

        Returns:
            Tokenizer instance
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Tokenizer loaded: {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {str(e)}")
            raise

    @staticmethod
    def _get_torch_dtype(dtype: str) -> torch.dtype:
        """
        Convert string dtype to torch dtype.

        Args:
            dtype: String representation of dtype

        Returns:
            torch.dtype
        """
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            logger.warning(f"Unknown dtype {dtype}, using fp32")
            return torch.float32
        return dtype_map[dtype]

    @staticmethod
    def _create_quantization_config(
        load_in_8bit: bool, load_in_4bit: bool
    ) -> Optional[BitsAndBytesConfig]:
        """
        Create BitsAndBytes quantization configuration.

        Args:
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization

        Returns:
            BitsAndBytesConfig or None
        """
        if load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)

        if load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        return None

    @staticmethod
    def _apply_lora(
        model, lora_config: Optional[Dict[str, Any]] = None
    ):
        """
        Apply LoRA fine-tuning adapter to model.

        Args:
            model: The model to apply LoRA to
            lora_config: LoRA configuration dictionary

        Returns:
            Model with LoRA adapter applied
        """
        if lora_config is None:
            lora_config = {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM,
            }

        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    **kwargs,
) -> tuple:
    """
    Convenience function to load a pretrained model and tokenizer.

    Args:
        model_name: HuggingFace model ID
        device: Device to load model on
        **kwargs: Additional arguments to pass to ModelLoader.load_pretrained_model

    Returns:
        tuple: (model, tokenizer)

    Example:
        >>> model, tokenizer = load_model_and_tokenizer(
        ...     "meta-llama/Llama-2-7b",
        ...     load_in_4bit=True,
        ...     use_lora=True,
        ... )
    """
    loader = ModelLoader(device=device)
    return loader.load_pretrained_model(model_name, **kwargs)
