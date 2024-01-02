from __future__ import annotations

import json
import random


def perturb_passage(passage, position=None, injection=None):
    """
    Perturbs a given passage by injecting a string at a specified position.

    Args:
        passage (str): The original passage to be perturbed.
        injection (str): The string to be injected into the passage.
        position (int or str): The position at which to inject the string. If "random", the string is injected at a random position.

    Returns:
        str: The perturbed passage with the injected string.
    """
    passage = str(passage)
    if position is None or injection is None:
        return passage

    sentences = passage.split(".")
    orig_position = position
    if orig_position == "random":
        position = random.randint(0, len(sentences) - 1)
    elif orig_position == -1:
        position = len(sentences)
    elif 0 <= orig_position < 1:
        position = int(position * len(sentences))

    sentences.insert(position, " " + injection)
    return ".".join(sentences)


def load_hackaprompt_injections(path):
    with open(path, "r", encoding="utf-8") as infile:
        prompt_injections = json.load(infile)

    return [(p["injection"], "I have been PWNED") for p in prompt_injections]
