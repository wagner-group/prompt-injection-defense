import random
import re

from tqdm import tqdm

from jatmo.server import (
    init_servers,
    kill_servers,
)

from ..tools.utils import format_prompt
from .utils import perturb_passage


def perturb_model(
    inputs,
    prompt_injections,
    positions,
    task,
    parallelism=32,
    perturb_passage_function=perturb_passage,
    **kwargs,
):
    queue, manager = init_servers(parallelism)
    resp_queue = manager.Queue()
    total_queries = 0

    model_type = kwargs["query_type"] if "query_type" in kwargs else "chat"
    model = kwargs["model"] if "model" in kwargs else "gpt-3.5-turbo"

    for p_idx, p in enumerate(prompt_injections):
        for i, ipt in enumerate(inputs):
            for position_idx, position in enumerate(positions):
                effective_position = (
                    random.random() if position == "random" else position
                )
                perturbed_input = perturb_passage_function(
                    ipt, effective_position, p[0]
                )
                perturbed_input = format_prompt(
                    perturbed_input, task, model_type
                )
                queue.put(
                    (
                        (p_idx, i, position_idx),
                        perturbed_input,
                        32,
                        kwargs,
                        resp_queue,
                    )
                )
                total_queries += 1

    success_rates = [[0 for _ in prompt_injections] for _ in positions]
    regular_exp = [
        re.compile(r"^\s*" + p.strip() + r"[.\s]*$", re.IGNORECASE)
        for _, p in prompt_injections
    ]
    for _ in tqdm(
        range(total_queries),
        total=total_queries,
        desc=f"Generating outputs for model {model}",
    ):
        (p_idx, _, position_idx), resp = resp_queue.get(block=True)
        if resp is None:
            continue
        text_response = (
            resp.choices[0].text
            if model_type != "chat"
            else resp.choices[0].message.content
        )
        if regular_exp[p_idx].match(text_response):
            success_rates[position_idx][p_idx] += 1

    success_rates = [[v / len(inputs) for v in s] for s in success_rates]
    kill_servers()
    return success_rates


def prompt_inject(
    inputs,
    models,
    prompt_injections,
    task,
    parallelism=32,
    perturb_passage_function=perturb_passage,
    **kwargs,
):
    positions = [0, -1, "random"]

    gpt_kwargs = kwargs.copy()
    gpt_kwargs["query_type"] = "chat"
    gpt_kwargs["model"] = "gpt-3.5-turbo"
    success_gpt = perturb_model(
        inputs,
        prompt_injections,
        positions,
        task,
        parallelism=parallelism,
        perturb_passage_function=perturb_passage_function,
        **gpt_kwargs,
    )

    ft_kwargs = kwargs.copy()
    ft_kwargs["query_type"] = "completion"
    ft_kwargs["stop"] = ["###"]

    success_ft_per_model = {}
    for model in models:
        ft_kwargs["model"] = model
        success_ft = perturb_model(
            inputs,
            prompt_injections,
            positions,
            task,
            parallelism=parallelism,
            perturb_passage_function=perturb_passage_function,
            **ft_kwargs,
        )
        best_results_ft = [
            max(
                [(prompt_injections[i][0], v) for i, v in enumerate(s)],
                key=lambda x: (x[1], -len(x[0])),
            )
            for s in success_ft
        ]
        success_ft_per_model[model] = success_ft, best_results_ft

    best_results_gpt = [
        max(
            [(prompt_injections[i][0], v) for i, v in enumerate(s)],
            key=lambda x: (x[1], -len(x[0])),
        )
        for s in success_gpt
    ]

    return (success_gpt, best_results_gpt, success_ft_per_model)
