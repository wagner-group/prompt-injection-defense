# from datasets import load_dataset

from jatmo import ConfigSpec, jatmo_synthetic

task_prompt = "Describe the following comment-free python function in a one-line master comment."


def run(training_set_sizes, path, parallelism=32):
    # First, load data
    # raw_inputs = wrapper(
    #     lambda: gather_inputs(
    #         total_count=200 + max(training_set_sizes),
    #         max_tokens=max_tokens,
    #     ),
    #     path,
    #     "raw_inputs.pkl",
    # )

    # Create config
    config = ConfigSpec()
    config.path = path
    config.training_set_sizes = training_set_sizes
    config.teacher = "gpt-3.5-turbo"
    config.parallelism = parallelism
    config.task = task_prompt
    config.oneshot = None
    config.no_formatting = True

    # Run
    _, config = jatmo_synthetic(
        config=config,
        print_results=True,
        evaluate=False,
    )
