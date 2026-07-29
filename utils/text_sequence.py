import random


SEPARATORS = {
    "comma": (",", ", "),
    "line break": (None, "\n"),
}


def randomize_text_sequence(text, separator, seed):
    """Return non-empty text items in a deterministic shuffled order."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if separator not in SEPARATORS:
        raise ValueError(f"Unknown separator: {separator}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    split_separator, output_separator = SEPARATORS[separator]
    raw_items = text.splitlines() if split_separator is None else text.split(split_separator)
    items = [item.strip() for item in raw_items if item.strip()]

    random.Random(seed).shuffle(items)
    return output_separator.join(items)
