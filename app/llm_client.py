"""
Thin provider-agnostic LLM client. The rest of the app (generator,
validator) calls `generate()` and doesn't need to know or care which
provider is behind it. Switch providers via LLM_PROVIDER in .env.

Default provider is Gemini because Google AI Studio's free tier requires
no credit card and no payment -- appropriate for a zero-budget student
project. Anthropic is kept as a drop-in alternative for later, once/if
you have paid credits.
"""

from app.config import settings


def generate(system: str, user_content: str, max_tokens: int = 500) -> str:
    if settings.llm_provider == "gemini":
        return _generate_gemini(system, user_content, max_tokens)
    elif settings.llm_provider == "anthropic":
        return _generate_anthropic(system, user_content, max_tokens)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}' -- expected "
            f"'gemini' or 'anthropic'."
        )


def _generate_gemini(system: str, user_content: str, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model_name,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.2,
        ),
    )
    return (response.text or "").strip()


def _generate_anthropic(system: str, user_content: str, max_tokens: int) -> str:
    from anthropic import Anthropic

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in your .env file.")

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.model_name,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()
