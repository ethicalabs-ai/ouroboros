import json
import os
import random

import click

from datasets import load_dataset


@click.command()
@click.option(
    "--dataset_name",
    required=True,
    help="Name of the Hugging Face dataset, e.g., tatsu-lab/alpaca",
)
@click.option(
    "--split",
    default="train",
    help="Split of the dataset to use (default: train)",
)
@click.option(
    "--instruction_field",
    default="instruction",
    help="Field name for the instruction in the dataset (default: instruction)",
)
@click.option(
    "--input_field",
    default="input",
    help="Field name for the input in the dataset (default: input)",
)
@click.option(
    "--prompts_file",
    required=True,
    type=click.Path(),
    help="Path to the JSON output file, e.g., prompts/nlp-alpaca.jsonl",
)
@click.option(
    "--shuffle",
    is_flag=True,
    help="Shuffle the dataset before processing (default: False)",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit the number of samples to process (default: all)",
)
def main(
    dataset_name,
    split,
    instruction_field,
    input_field,
    prompts_file,
    shuffle,
    limit,
):
    """Creates prompts from an existing Hugging Face dataset and saves them as a JSONL file."""

    if os.path.exists(prompts_file):
        if not click.confirm(
            f"{prompts_file} already exists. Overwrite?", default=False
        ):
            click.echo("Operation cancelled.")
            return

    # Load the dataset from Hugging Face
    ds = load_dataset(dataset_name, split=split)

    # Shuffle the dataset if the flag is set
    if shuffle:
        ds = ds.shuffle(seed=42)  # Ensuring reproducibility

    # Limit the number of samples if specified
    if limit:
        ds = ds.select(range(min(limit, len(ds))))  # Avoids out-of-range errors

    # Open the output file for writing
    with open(prompts_file, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = row.get(instruction_field, "").strip()
            inp = row.get(input_field, "").strip()

            prompt = f"{instruction}\n{inp}" if inp else instruction
            record = {
                "dataset_name": dataset_name,  # Add dataset handle
                "prompt": prompt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    click.echo(f"Prompts successfully saved to {prompts_file}")


if __name__ == "__main__":
    main()
