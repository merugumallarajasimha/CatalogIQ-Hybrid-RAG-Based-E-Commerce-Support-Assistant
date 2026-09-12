import os
from typing import Optional, List, Dict, Any
from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


_client: Optional[OpenAI] = None


def get_openrouter_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        _client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL
        )
    return _client


def generate_response(
    prompt: str,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None
) -> str:
    client = get_openrouter_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content


def generate_streaming(
    prompt: str,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None
):
    client = get_openrouter_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    print("OpenRouter client module")
    print("Set OPENROUTER_API_KEY environment variable to use.")
    print("This will be used in Phase 4 for LLM generation.")