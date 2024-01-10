"""
Automatic derivation of task specific fine-tuned models

Example tasks: 
    "Summarize groups of Amazon reviews into a single meta-review"
    "Summarize news articles into three sentence summaries."
    "Give a score between 0 and 5 indicating the similarity of two sentences."
    "Translate extracts from the Gutenberg Project from English to French"
    "Write a one-line comment describing the following python function without comments"
    "Classify the following movie review as positive or negative."
    "Determine whether the following comment is toxic."
"""

import argparse

from .code_summarization import run as run_code_summarization
from .news_summarization import run as run_news_summarization
from .review_summarization import run as run_review_summarization
from .translation import run as run_translation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("-f", "--few-shot", type=int, default=0)
    parser.add_argument(
        "--tss", "--training-set-sizes", type=int, nargs="+", default=[400]
    )
    parser.add_argument(
        "-a", "--additional-rules", type=str, nargs="+", default=None
    )
    args = parser.parse_args()

    task = args.task
    path = args.path
    training_set_sizes = args.tss

    if task == "review_summarization":
        run_review_summarization(
            training_set_sizes,
            path,
            fewshot=args.few_shot,
            additional_rules=args.additional_rules,
        )
    if task == "translation":
        run_translation(
            training_set_sizes,
            path,
            fewshot=args.few_shot,
            additional_rules=args.additional_rules,
        )
    if task == "news_summarization":
        run_news_summarization(
            training_set_sizes,
            path,
            fewshot=args.few_shot,
            additional_rules=args.additional_rules,
        )
    if task == "code_summarization":
        run_code_summarization(
            training_set_sizes,
            path,
            fewshot=args.few_shot,
            additional_rules=args.additional_rules,
        )
