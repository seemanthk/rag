"""
LLM handler module for managing multiple open-source LLMs
Supports Llama-3, Mistral, and Phi-3 with quantization
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMHandler:
    """Handler for loading and running open-source LLMs"""

    def __init__(self,
                 model_name: str,
                 quantization: str = "4bit",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 device: str = "auto"):
        """
        Initialize LLM handler

        Args:
            model_name: HuggingFace model name
            quantization: Quantization type ("4bit", "8bit", or None)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_model(self):
        """Load the model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Quantization: {self.quantization}")

        # Configure quantization
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device,
            "attn_implementation": "eager",  # Fix for Phi-3 compatibility
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
        except Exception as e:
            # Fallback without attn_implementation if it fails
            logger.warning(f"Failed with attn_implementation, trying without: {e}")
            model_kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

        logger.info("Model loaded successfully")

        # Create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Merge kwargs with defaults
        temp = kwargs.get("temperature", self.temperature)
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": max(temp, 0.1),  # Ensure temp >= 0.1 to avoid CUDA errors
            "top_p": kwargs.get("top_p", self.top_p),
            "use_cache": False,  # Disable cache to avoid compatibility issues
            "do_sample": True,
            "top_k": 50,  # Add top_k for better sampling
            "pad_token_id": self.tokenizer.eos_token_id,  # Explicit padding
        }

        # Generate with multiple fallback strategies
        try:
            outputs = self.pipeline(prompt, **gen_kwargs)
        except (AttributeError, RuntimeError) as e:
            error_msg = str(e)

            # Fallback 1: Cache error
            if "seen_tokens" in error_msg:
                logger.warning(f"Cache error, retrying without use_cache: {e}")
                gen_kwargs.pop("use_cache", None)
                outputs = self.pipeline(prompt, **gen_kwargs)

            # Fallback 2: CUDA error - skip this model
            elif "CUDA" in error_msg or "assert" in error_msg.lower() or "Accelerator" in error_msg:
                logger.error(f"CUDA error detected for {self.model_name}. Skipping this model.")
                logger.error(f"Error: {error_msg[:200]}")
                # Return a placeholder response
                return f"[ERROR: CUDA issue with {self.model_name}. Model skipped due to GPU compatibility issues. Try running with CPU or different model.]"
            else:
                raise

        # Extract generated text (remove prompt)
        generated_text = outputs[0]["generated_text"]
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts

        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """Get model information"""
        info = {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "device": self.device,
        }

        if self.model is not None:
            info["model_loaded"] = True
            info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
        else:
            info["model_loaded"] = False

        return info

    def unload_model(self):
        """Unload model from memory"""
        logger.info("Unloading model...")
        del self.model
        del self.tokenizer
        del self.pipeline
        self.model = None
        self.tokenizer = None
        self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded")


class MultiLLMManager:
    """Manager for handling multiple LLMs"""

    def __init__(self, llm_configs: Dict[str, Dict]):
        """
        Initialize multi-LLM manager

        Args:
            llm_configs: Dictionary mapping LLM names to config dicts
        """
        self.llm_configs = llm_configs
        self.llms = {}
        self.current_llm = None

    def load_llm(self, llm_name: str):
        """
        Load a specific LLM

        Args:
            llm_name: Name of LLM to load
        """
        if llm_name not in self.llm_configs:
            raise ValueError(f"Unknown LLM: {llm_name}")

        if llm_name in self.llms:
            logger.info(f"LLM {llm_name} already loaded")
            self.current_llm = llm_name
            return

        config = self.llm_configs[llm_name]

        llm = LLMHandler(
            model_name=config["model_name"],
            quantization=config.get("quantization", "4bit"),
            max_new_tokens=config.get("max_new_tokens", 512),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
        )

        llm.load_model()
        self.llms[llm_name] = llm
        self.current_llm = llm_name

        logger.info(f"Loaded LLM: {llm_name}")

    def unload_llm(self, llm_name: str):
        """
        Unload a specific LLM

        Args:
            llm_name: Name of LLM to unload
        """
        if llm_name in self.llms:
            self.llms[llm_name].unload_model()
            del self.llms[llm_name]
            if self.current_llm == llm_name:
                self.current_llm = None
            logger.info(f"Unloaded LLM: {llm_name}")

    def switch_llm(self, llm_name: str):
        """
        Switch to a different LLM (loads if not already loaded)

        Args:
            llm_name: Name of LLM to switch to
        """
        if llm_name not in self.llms:
            self.load_llm(llm_name)
        else:
            self.current_llm = llm_name
        logger.info(f"Switched to LLM: {llm_name}")

    def generate(self, prompt: str, llm_name: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using current or specified LLM

        Args:
            prompt: Input prompt
            llm_name: Optional LLM name (uses current if not specified)
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        llm_name = llm_name or self.current_llm

        if llm_name is None:
            raise ValueError("No LLM specified and no current LLM set")

        if llm_name not in self.llms:
            raise ValueError(f"LLM {llm_name} not loaded")

        return self.llms[llm_name].generate(prompt, **kwargs)

    def generate_all(self, prompt: str, **kwargs) -> Dict[str, str]:
        """
        Generate text using all loaded LLMs

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Dictionary mapping LLM names to generated texts
        """
        results = {}
        for llm_name, llm in self.llms.items():
            logger.info(f"Generating with {llm_name}...")
            results[llm_name] = llm.generate(prompt, **kwargs)
        return results

    def get_loaded_llms(self) -> List[str]:
        """Get list of loaded LLM names"""
        return list(self.llms.keys())
