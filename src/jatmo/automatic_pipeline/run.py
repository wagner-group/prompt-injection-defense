import os
import random
import string

import dill
import yaml

from ..tools import setup_dir, wrapper
from ..tools.eval_model import eval_model
from ..tools.finetune import finetune_model
from ..tools.output_generation import label_inputs
from ..tools.utils import (
    ConfigSpec,
)
from .eval_model import compare_to_ft_model
from .input_generation import format_inputs, get_input_list


def jatmo_synthetic_external_dataset_eval(
    task=None,
    path=None,
    orig_data=None,
    config=None,
    print_results=False,
):
    if config is None and task is None:
        raise ValueError("Must specify either config or task.")

    if config is not None and task is not None:
        raise ValueError("Must specify either config or task, not both.")

    if config is None and path is None:
        raise ValueError("Must specify either config or path.")

    if config is not None and path is not None:
        raise ValueError("Must specify either config or path, not both.")

    if config is None and orig_data is None:
        raise ValueError("Must specify either config or orig_data.")

    if config is None:
        config = ConfigSpec()
        config.path = path
        config.task = task

    elif isinstance(config, str):
        with open(config, "r", encoding="utf-8") as infile:
            config = ConfigSpec.from_dict(yaml.safe_load(infile))

    if config.models is None or not len(config.models):
        with open(
            os.path.join(path, "model_id.txt"), "r", encoding="utf-8"
        ) as f:
            config.models = [
                line.split("\t")[0].strip() for line in f.readlines()
            ]

    with open(os.path.join(config.path, "example.pkl"), "rb") as infile:
        example = dill.load(infile)

    if orig_data is not None:
        config.orig_data = orig_data

    if isinstance(config.orig_data, str):
        with open(config.orig_data, "rb") as infile:
            orig_data = dill.load(infile)
    else:
        orig_data = config.orig_data

    rslt = compare_to_ft_model(
        config.path,
        orig_data,
        example,
        config.models,
        config.task,
        parallelism=config.parallelism,
        redo_empty_responses=config.force,
        temperatures=config.temperatures,
        no_formatting=config.no_formatting,
    )

    if print_results:
        for t, v in rslt.items():
            for m, vv in v.items():
                print(f"Model {m} (temp {t}): {vv}")

    return rslt


def jatmo_synthetic(
    task=None,
    one_shot_example=None,
    config=None,
    print_results=False,
    evaluate=True,
    use_random_seed=True,
):
    if config is None and task is None:
        raise ValueError("Must specify either config or task.")

    if config is not None and task is not None:
        raise ValueError("Must specify either config or task, not both.")

    if config is not None and one_shot_example is not None:
        raise ValueError(
            "Must specify either config or one_shot_example, not both."
        )

    if config is None:
        config = ConfigSpec()
        config.path = "." + "".join(random.choices(string.ascii_lowercase, k=6))
        config.task = task
        config.training_set_sizes = [800]
        config.oneshot = one_shot_example

    if config.temperatures is None or not len(config.temperatures):
        config.temperatures = [1.0, 0.7]

    elif isinstance(config, str):
        with open(config, "r", encoding="utf-8") as infile:
            config = ConfigSpec.from_dict(yaml.safe_load(infile))

    setup_dir(config.path)

    path = config.path
    task = config.task
    additional_rules = config.rules
    max_training_set_size = max(config.training_set_sizes)

    train_ct, val_ct, test_ct = max_training_set_size, config.eval, config.test
    gen_ct = train_ct + val_ct + test_ct

    inputs = wrapper(
        lambda: get_input_list(
            task,
            gen_ct,
            additional_rules=additional_rules,
            example=config.oneshot,
            use_random_seed=use_random_seed,
        ),
        path,
        "raw_inputs.pkl",
    )

    gpt_inputs, ft_inputs, example = wrapper(
        lambda: format_inputs(task, inputs, example=config.oneshot),
        path,
        "formatted_inputs.pkl",
    )

    with open(os.path.join(path, "example.pkl"), "wb") as outfile:
        dill.dump(example, outfile)

    labels = wrapper(
        lambda: label_inputs(gpt_inputs, model="gpt-3.5-turbo"),
        path,
        "gpt_train_val_outputs.pkl",
    )

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

    for tss in config.training_set_sizes:
        real_train_ct = min(train_ct, tss)

        if real_train_ct in model_ids:
            continue

        model_id = finetune_model(
            path,
            (ft_inputs[:real_train_ct], labels[:real_train_ct]),
            (
                ft_inputs[train_ct : train_ct + val_ct],
                labels[train_ct : train_ct + val_ct],
            ),
        )

        with open(
            os.path.join(path, "model_id.txt"), "a", encoding="utf-8"
        ) as f:
            f.write(f"{model_id}\t{real_train_ct}\n")

        model_ids[real_train_ct] = model_id

        if evaluate:
            eval_results = eval_model(
                path,
                {
                    model_id: ft_inputs[
                        train_ct + val_ct : train_ct + val_ct + test_ct
                    ],
                    "gpt-3.5-turbo": gpt_inputs[
                        train_ct + val_ct : train_ct + val_ct + test_ct
                    ],
                },
                [model_id, "gpt-3.5-turbo"],
                gpt_inputs[train_ct + val_ct : train_ct + val_ct + test_ct],
                {
                    "gpt-3.5-turbo": labels[
                        train_ct + val_ct : train_ct + val_ct + test_ct
                    ]
                },
            )

            if print_results:
                print(
                    f"Eval results for training size of {real_train_ct}:\n\n"
                    + "\n".join(
                        f"{model}: {avg}" for model, avg in eval_results.items()
                    )
                )

    config.models = model_ids.values()

    return model_ids, config


def jatmo_synthetic_preview(tasks, ct=10, additional_rules=[]):
    if isinstance(tasks, str):
        tasks = [tasks]
        is_str = True
    else:
        is_str = False

    responses = [
        get_input_list(
            task,
            ct,
            additional_rules=additional_rules,
        )
        for task in tasks
    ]

    return responses[0] if is_str else responses
