# from datasets import load_dataset
import json

from jatmo import (
    ConfigSpec,
    jatmo_synthetic,
    jatmo_synthetic_external_dataset_eval,
)

task_prompt = "Describe the following comment-free python function in a one-line master comment."


def gather_inputs(total_count=100, path="data/code/code-dataset.json"):
    with open(path, encoding="utf-8") as infile:
        return [item["prompt"] for item in json.load(infile)][:total_count]


def run(
    training_set_sizes,
    path,
    parallelism=32,
    fewshot=0,
    additional_rules=None,
):
    # First, load data
    raw_inputs = gather_inputs(
        total_count=200 + max(training_set_sizes),
    )

    # Create config
    config = ConfigSpec()
    config.path = path
    config.training_set_sizes = training_set_sizes
    config.teacher = "gpt-3.5-turbo"
    config.parallelism = parallelism
    config.task = task_prompt
    config.fewshot = raw_inputs[:fewshot] if fewshot else None
    config.no_formatting = True
    config.rules = additional_rules

    # Run
    _, config = jatmo_synthetic(
        config=config,
        print_results=True,
        evaluate=False,
    )

    # Eval
    jatmo_synthetic_external_dataset_eval(
        orig_data=raw_inputs[: config.test],
        config=config,
        print_results=True,
    )
