"""Shared LLM utilities for v4 pipeline modules."""
from __future__ import annotations

import os
from openai import OpenAI

_DEFAULT_MODEL = os.environ.get("LLM_PLANNER_MODEL", "deepseek-v4-pro")


def get_llm_client() -> OpenAI | None:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
