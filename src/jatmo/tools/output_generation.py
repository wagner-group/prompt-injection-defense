""" Genreate outputs for a given list of inputs. """
import math

from tqdm import tqdm

from ..server import init_servers, kill_servers


def label_inputs(
    inputs, parallelism=8, max_tokens=math.inf, force=False, **kwargs
):
    """
    Generate outputs for a given list of inputs.

    Args:
        inputs (list): List of input strings.
        parallelism (int, optional): Number of parallel processes to use. Defaults to 8.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to math.inf.
        force (bool, optional): Rerun generation if output is empty. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        str: Generated outputs.

    Raises:
        ValueError: If generating more than one output at a time.

    """

    outputs = ["" for _ in inputs]
    queue, manager = init_servers(parallelism)
    resp_queue = manager.Queue()

    if "timeout" not in kwargs:
        kwargs["timeout"] = 30

    if "n" in kwargs and kwargs["n"] > 1:
        raise ValueError(
            "label_inputs only supports generating one output at a time."
        )

    for i, inp in enumerate(inputs):
        queue.put((i, inp, max_tokens, kwargs, resp_queue))

    pbar = tqdm(total=len(inputs), desc=f"Generating {kwargs['model']} outputs")
    done = set()

    while len(done) != len(inputs):
        for _ in range(len(inputs) - len(done)):
            idx, resp = resp_queue.get(block=True)
            if resp is None or resp == 0:
                pbar.update(1)
                done.add(idx)
                continue

            if len(resp.choices) == 1:
                candidate = (
                    resp.choices[0].message.content
                    if "query_type" not in kwargs
                    or kwargs["query_type"] == "chat"
                    else resp.choices[0].text
                ).strip()
                if candidate == "" and force:
                    local_kwargs = kwargs.copy()
                    local_kwargs["n"] = 10
                    queue.put(
                        (idx, inputs[idx], max_tokens, local_kwargs, resp_queue)
                    )
                else:
                    pbar.update(1)
                    done.add(idx)
                    outputs[idx] = candidate
            else:
                for choice in resp.choices:
                    content = (
                        choice.message.content
                        if "query_type" not in kwargs
                        or kwargs["query_type"] == "chat"
                        else choice.text
                    ).strip()
                    if content == "":
                        continue
                    outputs[idx] = content
                    break
                pbar.update(1)
                done.add(idx)

    kill_servers()
    return outputs
