import random

from datasets import load_dataset

from jatmo import (
    ConfigSpec,
    jatmo_synthetic,
    jatmo_synthetic_external_dataset_eval,
)
from jatmo.tools import wrapper

task_prompt = (
    "Summarize the following long-form news article in a short paragraph."
)


def load_articles():
    dataset = load_dataset("cnn_dailymail", "3.0.0")["train"]
    return dataset


def gather_inputs(
    total_count=1000,
):
    articles = load_articles()
    article_idx = random.sample(range(len(articles)), (total_count))
    article_list = [articles[i] for i in article_idx]

    inputs = [a["article"] for a in article_list]

    return inputs


def run(
    training_set_sizes,
    path,
    oneshot=False,
    parallelism=32,
):
    # First, load data
    raw_inputs = wrapper(
        lambda: gather_inputs(total_count=100),
        path,
        "raw_inputs_from_dataset.pkl",
    )

    # Create config
    config = ConfigSpec()
    config.path = path
    config.training_set_sizes = training_set_sizes
    config.teacher = "gpt-3.5-turbo"
    config.parallelism = parallelism
    config.task = task_prompt
    config.oneshot = raw_inputs[0] if oneshot else None
    config.no_formatting = True

    # Run
    _, config = jatmo_synthetic(
        config=config,
        print_results=True,
        evaluate=False,
    )

    # Eval
    jatmo_synthetic_external_dataset_eval(
        orig_data=raw_inputs,
        config=config,
        print_results=True,
    )
