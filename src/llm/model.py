import os

from src.config import OPENAI_MODEL, LLM_TEMPERATURE

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
    return response.content.strip()