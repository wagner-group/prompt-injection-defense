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

import sys

from .code_summarization import run as run_code_summarization
from .news_summarization import run as run_news_summarization
from .review_summarization import run as run_review_summarization
from .translation import run as run_translation


def main():
    task = sys.argv[1]
    path = sys.argv[2]
    if sys.argv[3] == "oneshot":
        one_shot = True
        training_set_sizes = (
            [int(x) for x in sys.argv[4:]] if len(sys.argv) > 4 else [800]
        )
    else:
        one_shot = False
        training_set_sizes = (
            [int(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [800]
        )

    if task == "review_summarization":
        run_review_summarization(training_set_sizes, path, oneshot=one_shot)
    if task == "translation":
        run_translation(training_set_sizes, path, oneshot=one_shot)
    if task == "news_summarization":
        run_news_summarization(training_set_sizes, path, oneshot=one_shot)
    if task == "code_summarization":
        run_code_summarization(training_set_sizes, path)
