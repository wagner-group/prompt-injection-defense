import sys

from .news_summarization import run as run_news_summarization
from .review_summarization import run as run_review_summarization
from .translation import run as run_translation


def main():
    task = sys.argv[1]
    path = sys.argv[2]
    training_set_sizes = (
        [int(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [400]
    )

    if task == "review_summarization":
        run_review_summarization(training_set_sizes, path)
    if task == "translation":
        run_translation(training_set_sizes, path)
    if task == "news_summarization":
        run_news_summarization(training_set_sizes, path)
