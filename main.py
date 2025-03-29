import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
from huggingface_hub import HfFolder
from openai import OpenAI
from outlines import generate, models
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from datasets import Dataset, load_dataset


class EvaluationMetrics(BaseModel):
    logical_consistency: int
    clarity: int
    depth: int
    accuracy: int
    context_alignment: int


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RecursiveAIExperiment:
    def __init__(
        self,
        model_name: str = "deepseek-r1:1.5b",
        critique_model_name: str = "qwen2.5:0.5b",
        iteration_limit: int = 3,
    ):
        self.prev_score = 0
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        self.iteration_limit = iteration_limit
        self.model_name = model_name
        self.embed_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # Local embedding model
        self.critique_model_name = critique_model_name
        self.critique_model = models.openai(
            critique_model_name,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    def generate_responses(self, prompt: str) -> List[str]:
        """Generate multiple candidate responses by making separate API requests."""
        logging.info(
            f"Generating {self.iteration_limit} responses with {self.model_name} for prompt: {prompt}"
        )

        responses = []
        for i in range(self.iteration_limit):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            if (
                response.choices
                and hasattr(response.choices[0], "message")
                and hasattr(response.choices[0].message, "content")
            ):
                generated_text = response.choices[0].message.content
                logging.info(f"Response {i+1}: {generated_text}")
                responses.append(generated_text)
            else:
                logging.warning(f"No response generated for request {i+1}")

        logging.info(f"Generated responses: {responses}")
        return responses

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for a given text."""
        return np.array(self.embed_model.encode(text, convert_to_numpy=True))

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def rank_responses(self, responses: list, query: str) -> list:
        """Rank responses based on semantic similarity to the query."""
        query_embedding = self.get_embedding(query)
        ranked_responses = sorted(
            responses,
            key=lambda resp: self.cosine_similarity(
                self.get_embedding(resp), query_embedding
            ),
            reverse=True,
        )
        return ranked_responses

    def generate_critique(self, critique_text: str) -> EvaluationMetrics:
        """Parse critique text and extract structured scores."""
        generator = generate.json(self.critique_model, EvaluationMetrics)
        raw_evaluation = generator(critique_text)
        return raw_evaluation

    def critique(self, response: str, query: str) -> Dict:
        """Evaluate multiple responses in one API call"""
        logging.info(
            f"Critiquing response with {self.critique_model_name} for query: {query}"
        )
        # Construct the critique prompt
        prompt = (
            f"Evaluate the response to the query '{query}'.\n"
            "Provide scores (1-10) for:\n"
            "1. Logical consistency\n2. Clarity\n3. Depth\n4. Accuracy\n5. Context alignment\n"
            "Output strictly as a JSON object.\n"
            "Do NOT exceed the range 1-10.\n"
            f"{response}\n\n"
        )
        # Send request to model
        critique_result = self.generate_critique(prompt)
        logging.info(f"Critique received: {critique_result}")
        return dict(critique_result)

    def refine_response(self, response: str, query: str) -> str:
        """Improve response based on critique"""
        logging.info(f"Refining response for query: {query}")
        refinement_prompt = f"Improve this response to '{query}':\n\nOriginal response: {response}\n\nMake it more logical, concise, and well-structured. Revised response:"
        refined_response = (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.3,
            )
            .choices[0]
            .message.content
        )
        logging.info(f"Refined response: {refined_response}")
        return refined_response

    def recursive_improvement(
        self, response: str, query: str, iteration: int = 0
    ) -> str:
        """Recursively refine response until threshold met."""
        logging.info(f"Recursive improvement iteration {iteration} for query: {query}")

        if iteration >= self.iteration_limit:
            return response

        critique = self.critique(response, query) or {}
        critique_values = list(critique.values())

        avg_score = (
            sum(critique_values) / len(critique_values) if critique_values else 0
        )
        logging.info(f"Average critique score: {avg_score}")

        if avg_score >= 8.5:
            logging.info(f"Threshold met. Returning response: {response}")
            return response

        if iteration > 0 and abs(avg_score - self.prev_score) < 0.3:
            logging.info("Minimal improvement detected. Stopping refinement early.")
            return response

        self.prev_score = avg_score  # Store score for next iteration
        improved = self.refine_response(response, query)
        return self.recursive_improvement(improved, query, iteration + 1)

    def run_experiment(self, query: str) -> Dict:
        """Runs the recursive AI experiment with improved ranking."""
        logging.info(f"Starting experiment for query: {query}")

        # Step 1: Generate initial responses
        candidates = self.generate_responses(query)

        if not candidates:
            logging.error("No responses generated. Exiting experiment.")
            return {
                "initial_responses": [],
                "ranked_responses": [],
                "final_response": "",
            }

        # Step 2: Rank responses based on semantic similarity
        ranked = self.rank_responses(candidates, query)

        # Step 3: Select the best response (highest-ranked) for recursive improvement
        best_response = ranked[0] if ranked else candidates[0]

        # Step 4: Refine the best response
        final_response = self.recursive_improvement(best_response, query)

        logging.info(f"Experiment completed. Final response: {final_response}")

        return {
            "initial_responses": candidates,
            "ranked_responses": ranked,
            "final_response": final_response,
        }


def extract_reasoning(response: str) -> List[str]:
    """Extract reasoning steps from <think> tags or deduce steps if absent."""
    reasoning_steps = []

    # If response contains structured reasoning in <think> tags, extract them
    if "<think>" in response and "</think>" in response:
        think_content = response.split("<think>")[1].split("</think>")[0].strip()
        reasoning_steps = [
            step.strip() for step in think_content.split("\n") if step.strip()
        ]

    # If no <think> section, fallback to heuristic breakdown
    elif response:
        reasoning_steps = response.split(". ")  # Simple split by periods
        reasoning_steps = [step.strip() for step in reasoning_steps if step]

    return reasoning_steps


def clean_response(response: str) -> str:
    """
    Removes reasoning enclosed within <think> </think> tags from the response.

    Returns:
        str: The cleaned response without the <think> sections.
    """
    cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned_response.strip()


def load_prompt_records(prompts_file: str):
    """
    Load prompt records from a file.

    If prompts_file ends with '.jsonl', it is assumed to be a JSONL file:
      each line is a JSON object containing at least a "prompt" key.
    Otherwise, it is treated as a plain text file:
      each non-empty line is treated as a prompt (string).
    """
    if prompts_file.endswith(".jsonl"):
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "prompt" not in record:
                    raise ValueError("JSON lines must contain a 'prompt' field.")
                yield record
    else:
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield {"prompt": line}


def run_experiment_on_prompts(
    domain: str,
    model_name: str,
    critique_model_name: str,
    iteration_limit: int,
    existing_dataset: Optional[Dataset] = None,
    force: bool = False,
    source_dataset_file: Optional[str] = None,
    source_dataset_hf: Optional[str] = None,
    prompt_field: str = "prompt",
    response_field: str = "response",
    dataset_name: Optional[str] = None,
) -> Dataset:
    """
    Runs the Ouroboros pipeline on prompts.

    If source_dataset_file or source_dataset_hf is provided, each record is assumed to contain a prompt
    and an existing response (using prompt_field and response_field). The experiment
    then refines the existing response.
    Otherwise, it loads prompts from prompts_file and generates responses from scratch.
    The final record includes the domain, source dataset info, and a dataset_name.
    """
    ai_experiment = RecursiveAIExperiment(
        model_name, critique_model_name, iteration_limit
    )

    # Create a dictionary mapping "input" -> full record for easy lookup
    existing_records = {}
    if existing_dataset is not None and "input" in existing_dataset.column_names:
        existing_records = {row["input"]: row for row in existing_dataset}

    records = []
    source_name = None
    if source_dataset_file:
        records = list(load_prompt_records(source_dataset_file))
        source_name = source_dataset_file
    elif source_dataset_hf:
        ds = load_dataset(source_dataset_hf)
        split = list(ds.keys())[0]
        records = ds[split]
        source_name = source_dataset_hf

    for i, record in enumerate(records, start=1):
        prompt = record.get(prompt_field)
        original_response = record.get(response_field)
        if prompt is None or original_response is None:
            logging.warning(f"Skipping record #{i} due to missing fields.")
            continue

        if prompt in existing_records and not force:
            logging.info(f"Skipping existing prompt: {prompt}")
            continue

        logging.info(f"Refining record #{i} for prompt: {prompt}")
        refined_response = ai_experiment.recursive_improvement(
            original_response, prompt
        )
        reasoning_steps = extract_reasoning(refined_response)
        new_entry = {
            "input": prompt,
            "original_response": original_response,
            "completion": clean_response(refined_response),
            "reasoning": reasoning_steps if reasoning_steps else None,
            "domain": domain,
            "source_dataset": source_name,
            "dataset_name": dataset_name,
        }
        for k, v in record.items():
            if k not in {prompt_field, response_field}:
                new_entry[k] = v

        existing_records[prompt] = new_entry

    updated_dataset = Dataset.from_list(list(existing_records.values()))
    return updated_dataset


@click.command()
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory to save generated datasets.",
)
@click.option(
    "--hf_dataset",
    type=str,
    help="Hugging Face dataset repository to update, e.g., my_user/my_dataset",
)
@click.option(
    "--model_name",
    type=str,
    default="deepseek-r1:1.5b",
    help="Model used for response generation.",
)
@click.option(
    "--critique_model_name",
    type=str,
    default="qwen2.5:0.5b",
    help="Model used for critique refinement.",
)
@click.option(
    "--num_iterations",
    type=int,
    default=5,
    help="Number of recursive refinement iterations.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force reprocessing even if the prompt already exists in the dataset.",
)
@click.option(
    "--push_to_hf",
    is_flag=True,
    help="Push the updated dataset back to Hugging Face (only once at the end).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Automatically confirm overwriting files without prompting.",
)
@click.option(
    "--source_dataset",
    type=click.Path(exists=True),
    help="Path to the source dataset file to use for refinement (JSONL or text).",
)
@click.option(
    "--source_dataset_hf",
    type=str,
    help="Hugging Face dataset path to use for refinement, e.g., user/dataset_name",
)
@click.option(
    "--prompt_field",
    type=str,
    default="prompt",
    help="Field name in the source dataset that contains the prompt.",
)
@click.option(
    "--response_field",
    type=str,
    default="response",
    help="Field name in the source dataset that contains the response.",
)
@click.option(
    "--dataset_name",
    type=str,
    required=True,
    help="Name to assign to the final dataset.",
)
@click.option(
    "--domain",
    type=str,
    default=None,
    help="Set the domain for the dataset.",
)
def main(
    output_dir,
    hf_dataset,
    model_name,
    critique_model_name,
    num_iterations,
    force,
    push_to_hf,
    yes,
    source_dataset,
    source_dataset_hf,
    prompt_field,
    response_field,
    dataset_name,
    domain,
):
    logging.basicConfig(level=logging.INFO)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing_dataset = None
    if hf_dataset:
        logging.info(f"Loading dataset from Hugging Face: {hf_dataset}")
        try:
            ds_dict = load_dataset(hf_dataset)
            first_split = list(ds_dict.keys())[0]
            existing_dataset = ds_dict[first_split]
            logging.info(f"Loaded '{hf_dataset}' with {len(existing_dataset)} rows.")
        except Exception as e:
            logging.warning(f"Failed to load dataset: {e}")
            existing_dataset = None

    merged_dataset = existing_dataset
    changes_detected = False

    if source_dataset or source_dataset_hf:
        current_domain = domain if domain else "default"
        logging.info(
            f"Processing source dataset: {source_dataset or source_dataset_hf} with domain: {current_domain}"
        )
        updated_dataset = run_experiment_on_prompts(
            domain=current_domain,
            model_name=model_name,
            critique_model_name=critique_model_name,
            iteration_limit=num_iterations,
            existing_dataset=merged_dataset,
            force=force,
            source_dataset_file=source_dataset,
            source_dataset_hf=source_dataset_hf,
            prompt_field=prompt_field,
            response_field=response_field,
            dataset_name=dataset_name,
        )
        changes_detected = True
        merged_dataset = updated_dataset

    if changes_detected and merged_dataset is not None:
        dataset_path_parquet = output_path / "ouroboros_dataset.parquet"
        dataset_path_json = output_path / "ouroboros_dataset.json"

        if (
            dataset_path_parquet.exists()
            and not yes
            and not click.confirm(
                f"{dataset_path_parquet} exists. Overwrite?", default=True
            )
        ):
            logging.info("User cancelled overwrite. Exiting.")
            return

        merged_dataset.to_parquet(str(dataset_path_parquet))
        merged_dataset.to_json(str(dataset_path_json))
        logging.info(
            f"Dataset saved locally to: {dataset_path_parquet} & {dataset_path_json}"
        )
    else:
        logging.info("No changes detected. Skipping local save.")

    if push_to_hf and hf_dataset and changes_detected and merged_dataset is not None:
        token = HfFolder.get_token()
        if not token:
            logging.error(
                "Hugging Face authentication token not found. Run `huggingface-cli login` first."
            )
            return
        logging.info(f"Pushing updated dataset to Hugging Face: {hf_dataset}")
        merged_dataset.push_to_hub(repo_id=hf_dataset, private=False)
        logging.info("Dataset successfully pushed to Hugging Face.")
    elif push_to_hf:
        logging.info("No new data. Skipping Hugging Face push.")


if __name__ == "__main__":
    main()
