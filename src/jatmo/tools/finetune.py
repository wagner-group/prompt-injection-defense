import json
import re
import time

from openai import OpenAI

from .utils import format_prompt


def finetune_model(path, training, validation, **kwargs):
    # Format data for fine-tuning.
    training_formatted = format_finetune_data(training[0], training[1])
    validation_formatted = format_finetune_data(validation[0], validation[1])

    # Save to file
    with open(path + "/finetune.jsonl", "w", encoding="utf-8") as outfile:
        for entry in training_formatted:
            json.dump(entry, outfile)
            outfile.write("\n")

    with open(path + "/finetune_val.jsonl", "w", encoding="utf-8") as outfile:
        for entry in validation_formatted:
            json.dump(entry, outfile)
            outfile.write("\n")

    # Load data to openai server
    client = OpenAI()
    file_id = client.files.create(
        file=open(path + "/finetune.jsonl", "rb"),
        purpose="fine-tune",
    ).id

    file_val_id = client.files.create(
        file=open(path + "/finetune_val.jsonl", "rb"),
        purpose="fine-tune",
    ).id

    ft_job = client.fine_tuning.jobs.create(
        training_file=file_id,
        validation_file=file_val_id,
        model="davinci-002",
        **kwargs,
    )

    while client.fine_tuning.jobs.retrieve(ft_job.id).status in [
        "validating_files",
        "queued",
        "running",
    ]:
        time.sleep(60)

    ft_job = client.fine_tuning.jobs.retrieve(ft_job.id)
    if ft_job.status != "succeeded":
        raise RuntimeError(ft_job.failure_reason)

    else:
        return ft_job.fine_tuned_model


def format_finetune_data(inputs, outputs):
    """
    Format data for fine-tuning.

    Args:
        inputs (List[str]): A list of inputs for the fine-tuned model.
        outputs (List[str]): A list of outputs for the fine-tuned model.

    Returns:
        List[Dict]: A list of dictionaries containing the inputs and outputs for the fine-tuned model.
    """
    return [
        {
            "prompt": format_prompt(inp, model_type="completion"),
            "completion": " "
            + re.sub(r"[\s\n\t]*###[\s\n\t]*", "", outputs[i].strip())
            + "###",
        }
        for i, inp in enumerate(inputs)
    ]
