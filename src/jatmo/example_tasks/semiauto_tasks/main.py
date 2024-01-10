import argparse
import sys

from .news_summarization import run as run_news_summarization
from .review_summarization import run as run_review_summarization
from .translation import run as run_translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        type=str,
        choices=["review_summarization", "translation", "news_summarization"],
    )
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--tss", "--training-set-sizes", type=int, nargs="+", default=[400]
    )
    args = parser.parse_args()
    task = args.task
    path = args.path
    training_set_sizes = args.tss

    if task == "review_summarization":
        run_review_summarization(training_set_sizes, path)
    if task == "translation":
        run_translation(training_set_sizes, path)
    if task == "news_summarization":
        run_news_summarization(training_set_sizes, path)
