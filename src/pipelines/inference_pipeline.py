"""
Inference Pipeline Module
Handles running inference on loaded language models with support for
batched generation, sampling strategies, and various decoding parameters.
"""

import logging
from typing import Optional, Union

import torch

from src.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Pipeline for running inference on a language model."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "fp32",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device: str = "auto",
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize InferencePipeline.

        Args:
            model_name: HuggingFace model ID or path to local model.
            dtype: Weight dtype ('fp32', 'fp16', 'bfloat16').
            load_in_8bit: Load model in 8-bit quantization.
            load_in_4bit: Load model in 4-bit quantization.
            device: Device to run inference on ('cuda', 'cpu', or 'auto').
            trust_remote_code: Allow custom modeling code from HuggingFace.
            cache_dir: Directory for caching downloaded models.
        """
        self.model_name = model_name

        loader = ModelLoader(device=device)
        self.model, self.tokenizer = loader.load_pretrained_model(
            model_name=model_name,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.model.eval()
        self.device = loader.device
        logger.info("InferencePipeline ready on device: %s", self.device)

    def run(
        self,
        prompts: Union[str, list[str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        skip_special_tokens: bool = True,
    ) -> Union[str, list[str]]:
        """
        Run inference on one or more prompts.

        Args:
            prompts: A single prompt string or a list of prompt strings.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k sampling filter size.
            repetition_penalty: Penalty for repeating tokens (1.0 = disabled).
            do_sample: Whether to use sampling; False uses greedy decoding.
            num_return_sequences: Number of sequences to return per prompt.
            skip_special_tokens: Strip special tokens from decoded output.

        Returns:
            A single decoded string when one prompt is given, otherwise a list
            of decoded strings (length = len(prompts) * num_return_sequences).
        """
        single = isinstance(prompts, str)
        if single:
            prompts = [prompts]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logger.info(
            "Running inference: %d prompt(s), max_new_tokens=%d",
            len(prompts),
            max_new_tokens,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
        decoded = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=skip_special_tokens
        )

        logger.info("Inference complete, %d sequence(s) generated", len(decoded))
        return decoded[0] if single and num_return_sequences == 1 else decoded
