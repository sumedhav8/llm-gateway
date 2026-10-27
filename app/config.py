"""Model settings, read from the .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

PLACEHOLDER_KEY = "your-api-key-here"


@dataclass(frozen=True)
class ModelConfig:
    tier: str       # "cheap" or "strong"
    base_url: str   # where the model's API lives
    api_key: str    # secret key (or "ollama" for the local model)
    name: str       # model name the provider expects

    @property
    def is_configured(self) -> bool:
        """False while the API key is missing or still the placeholder."""
        return bool(self.api_key) and self.api_key != PLACEHOLDER_KEY


def _load(tier, prefix, default_url, default_key, default_name):
    return ModelConfig(
        tier=tier,
        base_url=os.getenv(f"{prefix}_BASE_URL", default_url),
        api_key=os.getenv(f"{prefix}_API_KEY", default_key),
        name=os.getenv(f"{prefix}_MODEL", default_name),
    )


MODELS = {
    "cheap": _load("cheap", "CHEAP", "http://localhost:11434/v1", "ollama", "llama3.2"),
    "strong": _load("strong", "STRONG", "https://api.anthropic.com/v1", PLACEHOLDER_KEY, "claude-sonnet-5-5"),
}

# Upper limit on answer length. Keeps answers (and costs) predictable.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))