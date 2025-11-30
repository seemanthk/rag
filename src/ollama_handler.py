"""
Ollama LLM handler module for managing local LLMs
Supports phi3, llama3, gemma2, and other Ollama models
"""

import requests
import logging
from typing import List, Dict, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaHandler:
    """Handler for loading and running Ollama LLMs"""

    def __init__(self,
                 model_name: str,
                 base_url: str = "http://localhost:11434",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 timeout: int = 120):
        """
        Initialize Ollama handler

        Args:
            model_name: Ollama model name (e.g., 'phi3', 'llama3', 'gemma2')
            base_url: Ollama API base URL
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.is_loaded = False

    def load_model(self):
        """Check if model is available and pull if needed"""
        logger.info(f"Checking Ollama model: {self.model_name}")

        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models_data = response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]

            # Check if our model is available (handle tags like :latest)
            model_available = any(
                self.model_name in model or model.startswith(f"{self.model_name}:")
                for model in available_models
            )

            if not model_available:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                logger.info(f"Attempting to pull {self.model_name}...")
                self._pull_model()

            self.is_loaded = True
            logger.info(f"Model {self.model_name} ready!")

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Please ensure Ollama is running.")
            logger.error("Start Ollama with: ollama serve")
            raise RuntimeError("Ollama is not running. Please start it with 'ollama serve'")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _pull_model(self):
        """Pull model from Ollama"""
        logger.info(f"Pulling model {self.model_name} (this may take a few minutes)...")

        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=600,  # 10 minute timeout for model pull
                stream=True
            )
            response.raise_for_status()

            # Stream progress
            for line in response.iter_lines():
                if line:
                    logger.info(line.decode('utf-8'))

            logger.info(f"Model {self.model_name} pulled successfully!")

        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt using Ollama

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self.is_loaded:
            logger.warning("Model not explicitly loaded. Attempting generation anyway...")

        # Merge kwargs with defaults
        options = {
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "num_predict": kwargs.get("max_new_tokens", self.max_new_tokens),
        }

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options
        }

        try:
            start_time = time.time()
            logger.info(f"Generating with {self.model_name}...")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get('response', '')

            elapsed_time = time.time() - start_time
            logger.info(f"Generation completed in {elapsed_time:.2f}s")

            return generated_text.strip()

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout}s")
            return f"[ERROR: Generation timed out for {self.model_name}]"
        except requests.exceptions.ConnectionError:
            logger.error("Lost connection to Ollama")
            return f"[ERROR: Cannot connect to Ollama. Please ensure it's running.]"
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"[ERROR: {str(e)}]"

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
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results

    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            response.raise_for_status()
            model_info = response.json()

            return {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "model_loaded": self.is_loaded,
                "model_details": model_info
            }
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
            return {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "model_loaded": self.is_loaded,
            }

    def unload_model(self):
        """Unload model from memory (optional for Ollama)"""
        logger.info(f"Marking model {self.model_name} as unloaded")
        self.is_loaded = False


class OllamaMultiLLMManager:
    """Manager for handling multiple Ollama LLMs"""

    def __init__(self, llm_configs: Dict[str, Dict], base_url: str = "http://localhost:11434"):
        """
        Initialize multi-LLM manager for Ollama

        Args:
            llm_configs: Dictionary mapping LLM names to config dicts
            base_url: Ollama API base URL
        """
        self.llm_configs = llm_configs
        self.base_url = base_url
        self.llms = {}
        self.current_llm = None

    def load_llm(self, llm_name: str):
        """Load an Ollama LLM"""
        if llm_name not in self.llm_configs:
            raise ValueError(f"Unknown LLM: {llm_name}")

        if llm_name in self.llms:
            logger.info(f"LLM {llm_name} already loaded")
            self.current_llm = llm_name
            return

        config = self.llm_configs[llm_name]

        llm = OllamaHandler(
            model_name=config["model_name"],
            base_url=config.get("base_url", self.base_url),
            max_new_tokens=config.get("max_new_tokens", 512),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.9),
            timeout=config.get("timeout", 120),
        )

        llm.load_model()
        self.llms[llm_name] = llm
        self.current_llm = llm_name

        logger.info(f"Loaded Ollama LLM: {llm_name}")

    def unload_llm(self, llm_name: str):
        """Unload a specific LLM"""
        if llm_name in self.llms:
            self.llms[llm_name].unload_model()
            del self.llms[llm_name]
            if self.current_llm == llm_name:
                self.current_llm = None
            logger.info(f"Unloaded LLM: {llm_name}")

    def switch_llm(self, llm_name: str):
        """Switch to a different LLM (loads if not already loaded)"""
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
            raise ValueError(f"LLM {llm_name} not loaded. Call load_llm('{llm_name}') first.")

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

    def load_all_llms(self):
        """Load all configured LLMs"""
        logger.info(f"Loading all {len(self.llm_configs)} LLMs...")
        for llm_name in self.llm_configs.keys():
            try:
                self.load_llm(llm_name)
            except Exception as e:
                logger.error(f"Failed to load {llm_name}: {e}")
        logger.info(f"Loaded {len(self.llms)} out of {len(self.llm_configs)} LLMs")
