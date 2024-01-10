from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import dacite
import dill


@dataclass(frozen=False)
class ConfigSpec:
    """Configuration spec"""

    path: str = ""
    task: str = ""
    teacher: str = "gpt-3.5-turbo"
    rules: Optional[List[str]] = field(default_factory=list, hash=False)
    fewshot: Optional[List[str]] = field(default_factory=list, hash=False)
    training_set_sizes: Optional[List[int]] = field(
        default_factory=list, hash=False
    )

    eval: int = 50
    test: int = 100
    parallelism: int = 8

    orig_data: Optional[Union[str, List[str]]] = None
    force: bool = True
    temperatures: Optional[List[float]] = field(
        default_factory=list, hash=False
    )
    models: List[str] = field(default_factory=list, hash=False)
    no_formatting: bool = False
    prompt_injections: List[str] = field(default_factory=list, hash=False)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> ConfigSpec:
        return dacite.from_dict(ConfigSpec, d)

    def set_prompt_injections(self, prompts, expected_responses):
        if len(prompts) != len(expected_responses):
            raise ValueError(
                "Prompts and expected responses must be the same length."
            )

        self.prompt_injections.extend(
            (p, r) for p, r in zip(prompts, expected_responses)
        )


def format_prompt(inputs, task=None, model_type="chat"):
    """
    Format data for passing to fine-tuned model.

    Args:
        inputs (List[str]): A list of inputs for the fine-tuned model.

    Returns:
        List[Dict]: A list of dictionaries containing the formatted inputs for the fine-tuned model.
    """
    if task is None and model_type == "chat":
        raise ValueError("Task must be specified for chat model.")

    if isinstance(inputs, str):
        inputs = [inputs]
        converted_from_str = True
    else:
        converted_from_str = False

    if model_type == "chat":
        rtn = [
            task + "\n\n" + re.sub(r"[\s\n\t]*###[\s\n\t]*$", "", inp.strip())
            for inp in inputs
        ]
    else:
        rtn = [
            re.sub(r"[\s\n\t]*###[\s\n\t]*$", "", inp.strip()) + "\n\n###"
            for inp in inputs
        ]

    return rtn[0] if converted_from_str else rtn


def wrapper(function_call, path, filename, force=False):
    if os.path.exists(os.path.join(path, filename)) and not force:
        with open(os.path.join(path, filename), "rb") as f:
            rtn = dill.load(f)
    else:
        rtn = function_call()
        with open(os.path.join(path, filename), "wb") as f:
            dill.dump(rtn, f)
    return rtn


def setup_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
