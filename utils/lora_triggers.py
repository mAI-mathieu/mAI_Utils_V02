"""Literal prompt matching without ComfyUI dependencies."""

import re


def prompt_matches_triggers(prompt: str, trigger_words: str) -> bool:
    """Match any comma/newline-separated word or phrase, ignoring case.

    Whitespace within phrases is flexible. Word boundaries prevent a trigger
    such as ``cat`` from activating on ``cathedral``. Punctuation is literal.
    """
    normalized_prompt = " ".join(prompt.casefold().split())
    for entry in re.split(r"[,\r\n]+", trigger_words):
        trigger = " ".join(entry.casefold().split())
        if trigger and re.search(r"(?<!\w)" + re.escape(trigger) + r"(?!\w)", normalized_prompt):
            return True
    return False
