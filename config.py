"""
╔══════════════════════════════════════════════════════════════╗
║  SHARED CONFIG — Change the model here, all modules update   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Want to switch models? Just change MODEL_NAME below:        ║
║                                                              ║
║  OpenRouter models (via API key):                            ║
║    "google/gemini-2.0-flash-001"     — fast, cheap           ║
║    "google/gemini-2.5-pro-preview"   — smartest Gemini       ║
║    "anthropic/claude-sonnet-4"    — great for code           ║
║    "openai/gpt-4o"                   — OpenAI flagship       ║
║    "meta-llama/llama-4-maverick"     — open source           ║
║    "deepseek/deepseek-chat-v3-0324"  — budget-friendly       ║
║                                                              ║
║  Browse all: https://openrouter.ai/models                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ════════════════════════════════════════════════════════════════
#  CHANGE THESE TO SWITCH MODELS — one place, all modules update
# ════════════════════════════════════════════════════════════════

MODEL_NAME = "google/gemini-2.0-flash-001"
TEMPERATURE = 0.7
API_BASE = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# ════════════════════════════════════════════════════════════════


def get_llm(temperature=None):
    """Get a configured LLM instance. All modules use this."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature if temperature is not None else TEMPERATURE,
        openai_api_base=API_BASE,
        openai_api_key=API_KEY,
    )