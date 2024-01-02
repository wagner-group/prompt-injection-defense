import os
import random
import string

import yaml

from ..tools import setup_dir, wrapper
from ..tools.eval_model import eval_model
from ..tools.finetune import finetune_model
from ..tools.output_generation import label_inputs
from ..tools.utils import ConfigSpec, format_prompt
from .perturb import prompt_inject
from .utils import perturb_passage


def jatmo(
    inputs,
    task=None,
    prompt_injections=None,
    config=None,
    custom_perturb_passage=None,
    print_results=False,
    evaluate=True,
    only_prompt_inject_teacher=False,
):
    if config is None and task is None:
        raise ValueError("Must specify either config or task.")

    if config is not None and task is not None:
        raise ValueError("Must specify either config or task, not both.")

    if config is not None and prompt_injections is not None:
        raise ValueError(
            "Must specify either config or prompt_injections, not both."
        )

    if config is None:
        config = ConfigSpec()
        config.path = "." + "".join(random.choices(string.ascii_lowercase, k=6))
        config.task = task
        config.training_set_sizes = [400]
        config.prompt_injections = prompt_injections

    elif isinstance(config, str):
        with open(config, "r", encoding="utf-8") as infile:
            config = ConfigSpec.from_dict(yaml.safe_load(infile))

    setup_dir(config.path)
    if custom_perturb_passage is None:
        custom_perturb_passage = perturb_passage

    # Generate outputs
    finetune_formatted_inputs = [custom_perturb_passage(ipt) for ipt in inputs]
    formatted_inputs = format_prompt(
        finetune_formatted_inputs,
        config.task,
        "chat",
    )
    outputs = wrapper(
        lambda: label_inputs(
            formatted_inputs,
            model=config.teacher,
            parallelism=config.parallelism,
        ),
        config.path,
        "outputs.pkl",
    )

    # Fine-tune models
    if os.path.exists(os.path.join(config.path, "model_id.txt")):
        with open(
            os.path.join(config.path, "model_id.txt"), "r", encoding="utf-8"
        ) as f:
            model_ids = {
                int(line.split("\t")[1]): line.split("\t")[0]
                for line in f.read().split("\n")
                if line
            }

    else:
        with open(
            os.path.join(config.path, "model_id.txt"), "w", encoding="utf-8"
        ) as f:
            pass

        model_ids = {}
        for training_set_size in config.training_set_sizes:
            if training_set_size + config.eval + config.test > len(
                finetune_formatted_inputs
            ):
                continue

            model = finetune_model(
                config.path,
                (
                    finetune_formatted_inputs[:training_set_size],
                    outputs[:training_set_size],
                ),
                (
                    finetune_formatted_inputs[
                        -config.eval - config.test : -config.test
                    ],
                    outputs[-config.eval - config.test : -config.test],
                ),
            )

            model_ids[training_set_size] = model
            with open(
                os.path.join(config.path, "model_id.txt"), "a", encoding="utf-8"
            ) as f:
                f.write(f"{model}\t{training_set_size}\n")

    config.models = model_ids.values()

    if not evaluate:
        return model_ids

    # Eval models
    ft_test_inputs = finetune_formatted_inputs[-config.test :]
    gpt_test_inputs = formatted_inputs[-config.test :]

    inputs_per_model = {model: ft_test_inputs for model in model_ids.values()}
    inputs_per_model[config.teacher] = gpt_test_inputs

    outputs_per_model = {config.teacher: outputs[-config.test :]}

    eval_output = wrapper(
        lambda: eval_model(
            config.path,
            inputs_per_model,
            list(model_ids.values()) + [config.teacher],
            gpt_test_inputs,
            outputs_per_model,
            parallelism=config.parallelism,
        ),
        config.path,
        "evaluation.pkl",
    )

    if print_results:
        inv_model_ids = {v: k for k, v in model_ids.items()}
        for model in eval_output:
            if model in inv_model_ids:
                print(
                    f"{model} (trained on {inv_model_ids[model]} samples): {eval_output[model]}"
                )
            else:
                print(f"{model}: {eval_output[model]}")

    if not config.prompt_injections:
        return model_ids, eval_output

    # Prompt injection eval
    if not only_prompt_inject_teacher:
        success_gpt, best_results_gpt, success_ft = wrapper(
            lambda: prompt_inject(
                inputs[-config.test :],
                config.models,
                config.prompt_injections,
                config.task,
                parallelism=config.parallelism,
                perturb_passage_function=custom_perturb_passage,
            ),
            config.path,
            "prompt_injection_results.pkl",
        )
    else:
        success_gpt, best_results_gpt, success_ft = prompt_inject(
            inputs[-config.test :],
            [],
            config.prompt_injections,
            config.task,
            parallelism=config.parallelism,
            perturb_passage_function=custom_perturb_passage,
        )

    if print_results:
        print("Best results GPT:")
        positions = [0, -1, "random"]
        for position, result in enumerate(best_results_gpt):
            print(f"At position {positions[position]}: {result[1]}")

        for model, val in success_ft.items():
            print(
                f"Best results {model} (trained on {inv_model_ids[model]} samples):"
            )
            for position, result in enumerate(val[1]):
                print(f"At position {positions[position]}: {result[1]}")

    return (
        model_ids,
        eval_output,
        (success_gpt, best_results_gpt, success_ft),
    )
