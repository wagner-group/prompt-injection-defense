import dill

from ..server import init_servers, kill_servers, rate_completions
from .finetune import (
    format_finetune_data,
)
from .output_generation import label_inputs


def eval_model(
    path,
    inputs_per_model,
    model_list,
    eval_inputs,
    outputs_per_model=None,
    parallelism=8,
    **kwargs,
):
    if not all(
        len(i) == len(j) and len(i) == len(eval_inputs)
        for i in inputs_per_model.values()
        for j in inputs_per_model.values()
    ):
        raise ValueError("Each model must have the same number of inputs.")

    if outputs_per_model is None:
        outputs_per_model = {model: [] for model in model_list}
    for model in model_list:
        if model not in outputs_per_model:
            outputs_per_model[model] = []
        elif len(outputs_per_model[model]) != len(eval_inputs):
            raise ValueError(
                "Each model must have the same number of outputs as inputs."
            )

    for model in model_list:
        if len(outputs_per_model[model]) == len(eval_inputs):
            continue

        if "ft" in model.lower():
            kwargs["query_type"] = "completion"
            kwargs["stop"] = ["###"]
        else:
            kwargs["query_type"] = "chat"

        kwargs["model"] = model
        kwargs["timeout"] = 30

        inputs = inputs_per_model[model]
        if "ft" in model.lower():
            inputs = [
                f["prompt"]
                for f in format_finetune_data(inputs, ["None" for _ in inputs])
            ]

        outputs_per_model[model] = label_inputs(
            inputs,
            parallelism=parallelism,
            max_tokens=2048,
            force=False,
            **kwargs,
        )

    # Rate completions
    prompts = []
    responses = []

    for model in model_list:
        prompts += eval_inputs
        responses += outputs_per_model[model]

    queue, mgr = init_servers(parallelism)
    resp_queue = mgr.Queue()
    ratings = rate_completions(prompts, responses, queue, resp_queue)
    kill_servers()

    with open(path + "/eval_outputs.pkl", "wb") as outfile:
        dill.dump((eval_inputs, inputs_per_model, outputs_per_model), outfile)

    with open(path + "/eval_ratings.tsv", "w", encoding="utf-8") as outfile:
        outfile.write("index\t" + "\t".join(model_list) + "\n")
        for i, _ in enumerate(eval_inputs):
            line = (
                f"{i}\t"
                + "\t".join(
                    str(ratings[i + j * len(eval_inputs)])
                    for j, _ in enumerate(model_list)
                )
                + "\n"
            )
            outfile.write(line)
    return {
        m: sum(ratings[j * len(eval_inputs) : (j + 1) * len(eval_inputs)])
        / len(eval_inputs)
        for j, m in enumerate(model_list)
    }
