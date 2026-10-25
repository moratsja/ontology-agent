import os
import re

from src.config import LLM_TEMPERATURE, OPENAI_MODEL

SESSION_TOKEN_USAGE = {"input_tokens": 0, "output_tokens": 0}

def get_chat_model():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in .env.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=OPENAI_MODEL, temperature=LLM_TEMPERATURE)

def complete(system, user):
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    
    response = get_chat_model().invoke(messages)
    
    metadata = getattr(response, "response_metadata", {}) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    
    if not token_usage and hasattr(response, "usage_metadata") and response.usage_metadata:
        token_usage = response.usage_metadata

    inp = token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
    out = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
    
    SESSION_TOKEN_USAGE["input_tokens"] += inp
    SESSION_TOKEN_USAGE["output_tokens"] += out

    content = response.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        content = "".join(parts)
    return _strip_fence(str(content))

# Retrieves and resets token usage for the current session
def get_and_reset_session_tokens():
    tokens = SESSION_TOKEN_USAGE.copy()
    SESSION_TOKEN_USAGE["input_tokens"] = 0
    SESSION_TOKEN_USAGE["output_tokens"] = 0
    return tokens

def _strip_fence(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
    return stripped.strip()