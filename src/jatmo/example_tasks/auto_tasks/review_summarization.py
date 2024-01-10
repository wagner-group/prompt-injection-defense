import json
import os
import random

import tiktoken

from jatmo import (
    ConfigSpec,
    jatmo_synthetic,
    jatmo_synthetic_external_dataset_eval,
)
from jatmo.tools import wrapper

task_prompt = (
    "Write a single summary paragraph summarizing all customer reviews."
)


def load_reviews(path):
    """
    Load and parse Amazon reviews from a file.

    Args:
        path (str): The path to the file containing the reviews.

    Returns:
        Tuple[Dict[str, List[Dict]], List[Dict]]: A tuple containing two elements:
            - A dictionary mapping product IDs to lists of reviews for that product.
            - A list of products with more than 100 reviews.
    """
    with open(path, encoding="utf-8") as infile:
        data = [line for line in infile.read().split("\n") if len(line)]
    data_parsed = []
    for line in data:
        try:
            data_parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    per_product = {d["asin"]: [] for d in data_parsed}
    for d in data_parsed:
        per_product[d["asin"]].append(d)
    product_pool = [p for p, val in per_product.items() if len(val) > 100]

    return per_product, product_pool


def write_prompt(
    reviews,
    position=None,
    injection=None,
):
    """
    Generate a prompt for summarizing customer reviews of a product.

    Args:
        products (dict): A dictionary mapping ASINs to lists of product review dictionaries.
        asin (str): The ASIN of the product to summarize reviews for.
        count (int, optional): The number of reviews to include in the prompt. Defaults to 10.
        perturb (bool, optional): Whether to include a perturbation message in the reviews. Defaults to False.
        perturb_message (str, optional): The perturbation message to include. Required if perturb is True.
        perturb_position (int, optional): The index at which to insert the perturbation message. Required if perturb is True.

    Returns:
        str: A prompt for summarizing the specified product's reviews.
    """
    count = len(reviews)
    reviews = reviews[:]

    if position is not None:
        if position == "random":
            real_position = random.randint(0, count - 1)
        elif position == -1:
            real_position = count - 1
        elif 0 <= position < 1:
            real_position = int(position * count)

        reviews[real_position] = injection

    reviews_formatted = "\n\n".join(
        ["Review #{}:\n{}".format(1 + i, d) for i, d in enumerate(reviews)]
    )

    return reviews_formatted


def gather_inputs(
    path,
    total_count=1000,
    review_count=10,
):
    products, product_pool = load_reviews(path)
    product_list = random.sample(product_pool, total_count)

    encoder = tiktoken.encoding_for_model("davinci-002")

    raw_inputs = []

    for asin in product_list:
        for _ in range(100):
            # Randomly sample reviews from the product
            reviews = random.sample(
                [p["reviewText"] for p in products[asin] if "reviewText" in p],
                review_count,
            )
            if len(encoder.encode(task_prompt + write_prompt(reviews))) > (
                4096 - 256
            ):
                continue
            else:
                break

        raw_inputs.append(reviews)

    return raw_inputs


def run(
    training_set_sizes,
    path,
    fewshot=0,
    parallelism=32,
    additional_rules=None,
):
    # First, load data
    raw_inputs = wrapper(
        lambda: gather_inputs(
            os.path.join(path, "reviews.json"),
            total_count=200 + max(training_set_sizes),
            review_count=10,
        ),
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
        orig_data=raw_inputs,
        config=config,
        print_results=True,
    )
