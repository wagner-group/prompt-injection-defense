import os
import re

import dill
from tqdm import tqdm

from ..server import init_servers, kill_servers, rate_completions
from ..tools.finetune import format_finetune_data
from ..tools.output_generation import label_inputs
from .utils import reformat_prompt


def compare_to_ft_model(
    path,
    inputs,
    example,
    model_ids,
    task,
    parallelism=8,
    redo_empty_responses=True,
    temperatures=None,
    no_formatting=False,
    **kwargs,
):
    if isinstance(model_ids, str):
        model_ids = [model_ids]

    if not isinstance(model_ids, list):
        model_ids = list(model_ids)

    orig_kwargs = kwargs.copy()
    if os.path.exists(os.path.join(path, "eval_formatting_output.pkl")):
        with open(
            os.path.join(path, "eval_formatting_output.pkl"), "rb"
        ) as infile:
            GPT_inputs, FT_inputs = dill.load(infile)
    elif no_formatting:
        FT_inputs = [
            format_finetune_data([t.strip()], [""])[0]["prompt"] for t in inputs
        ]
        GPT_inputs = [task + "\n###\n" + f for f in FT_inputs]
    else:
        # Format inputs
        queue, manager = init_servers(parallelism)
        resp_queue = manager.Queue()

        kwargs_reformat = kwargs.copy()
        kwargs_reformat["query_type"] = "chat"
        kwargs_reformat["model"] = "gpt-4-1106-preview"
        kwargs_reformat["temperature"] = 0
        kwargs_reformat["timeout"] = 60

        formatted_inputs = ["" for _ in inputs]
        prompt_inputs = [reformat_prompt(example, input) for input in inputs]
        for i, ipt in enumerate(prompt_inputs):
            queue.put(
                (
                    i,
                    ipt,
                    4096,
                    kwargs_reformat,
                    resp_queue,
                )
            )

        for _ in tqdm(
            inputs,
            total=len(inputs),
            desc="Reformatting inputs",
        ):
            p_idx, resp = resp_queue.get(block=True)
            try:
                resp_text = re.sub(
                    r"([\s\n\t]*START)|(END[\s\n\t]*)",
                    "",
                    resp.choices[0].message.content,
                ).strip()
                formatted_inputs[p_idx] = format_finetune_data(
                    [resp_text], [""]
                )[0]["prompt"]
                if p_idx % 20 == 0:
                    print(formatted_inputs[p_idx])
            except AttributeError:
                continue

        FT_inputs = [f + " " for f in formatted_inputs]
        GPT_inputs = [task + "\n###\n" + f for f in FT_inputs]

        kill_servers()

        with open(
            os.path.join(path, "eval_formatting_output.pkl"), "wb"
        ) as outfile:
            dill.dump((GPT_inputs, FT_inputs), outfile)

    rtn = {}
    if temperatures is None:
        temperatures = [1.0]

    for temp in temperatures:
        kwargs["temperature"] = temp
        # Generate GPT Outputs
        orig_kwargs["model"] = "gpt-3.5-turbo"
        orig_kwargs["temperature"] = temp
        GPT_outputs = label_inputs(
            GPT_inputs,
            max_tokens=512,
            force=redo_empty_responses,
            **orig_kwargs,
        )

        # Generate FT outputs
        outputs = {m: ["" for _ in FT_inputs] for m in model_ids}
        for model in model_ids:
            kwargs["query_type"] = "completion"
            kwargs["stop"] = ["###"]
            kwargs["model"] = model
            kwargs["timeout"] = 30

            outputs[model] = label_inputs(
                FT_inputs,
                max_tokens=512,
                force=redo_empty_responses,
                **kwargs,
            )

        # Rate completions
        prompts = GPT_inputs[:]
        responses = GPT_outputs[:]

        for model in model_ids:
            prompts += GPT_inputs[:]
            responses += outputs[model][:]

        with open(os.path.join(path, f"save_{temp}.pkl"), "wb") as outfile:
            dill.dump((GPT_outputs, outputs), outfile)

        queue, manager = init_servers(parallelism)
        resp_queue = manager.Queue()

        ratings = rate_completions(prompts, responses, queue, resp_queue)

        with open(
            path + f"/eval_ft_compare_outputs_{temp}.pkl", "wb"
        ) as outfile:
            dill.dump(
                (inputs, FT_inputs, GPT_inputs, GPT_outputs, outputs), outfile
            )

        with open(
            path + f"/eval_ratings_{temp}.tsv", "w", encoding="utf-8"
        ) as outfile:
            outfile.write("index\tGPT\t" + "\t".join(model_ids) + "\n")
            for i, _ in enumerate(inputs):
                line = (
                    f"{i}\t"
                    + "\t".join(
                        str(ratings[i + j * len(inputs)])
                        for j in range(len(model_ids) + 1)
                    )
                    + "\n"
                )
                outfile.write(line)

        rtn[temp] = {
            m: sum(ratings[j * len(inputs) : (j + 1) * len(inputs)])
            / len(inputs)
            for j, m in enumerate(["GPT"] + model_ids)
        }

    kill_servers()
    return rtn
