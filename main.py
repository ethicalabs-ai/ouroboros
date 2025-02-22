import logging
from pathlib import Path
from typing import Dict, List

import click
import numpy as np
from openai import OpenAI
from outlines import generate, models
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from datasets import Dataset


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

        # Step 2: Rank responses based on critique scores
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


def run_experiment_on_prompts(
    prompts_file: str,
    model_name: str,
    critique_model_name: str,
    iteration_limit: int = 5,
) -> Dataset:
    """Run RecursiveAIExperiment on all prompts and create a Hugging Face dataset."""
    ai_experiment = RecursiveAIExperiment(
        model_name=model_name,
        critique_model_name=critique_model_name,
        iteration_limit=iteration_limit,
    )

    # Load prompts from file
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    dataset_entries = []

    for i, prompt in enumerate(prompts):
        logging.info(f"Running experiment for prompt #{i + 1}: {prompt}")

        # Run experiment
        result = ai_experiment.run_experiment(prompt)

        # Extract structured reasoning
        reasoning_steps = extract_reasoning(result["final_response"])

        # Construct dataset entry
        dataset_entry = {
            "input": prompt,
            "reasoning": reasoning_steps if reasoning_steps else None,
            "completion": result["final_response"],
            "refinements": result["ranked_responses"],  # Store all refinements
        }

        dataset_entries.append(dataset_entry)

    # Create Hugging Face dataset
    dataset = Dataset.from_list(dataset_entries)
    return dataset


@click.command()
@click.option(
    "--prompt_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory containing prompt files, categorized by domain.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory to save generated datasets.",
)
@click.option(
    "--model_name",
    type=str,
    default="deepseek-r1:1.5b",
    help="Name of the model used for response generation.",
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
def main(prompt_dir, output_dir, model_name, critique_model_name, num_iterations):
    logging.basicConfig(level=logging.INFO)

    prompt_path = Path(prompt_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    domains = [
        f.stem for f in prompt_path.glob("*.txt")
    ]  # Extract domain names from filenames

    for domain in domains:
        logging.info(f"Processing domain: {domain}")

        dataset = run_experiment_on_prompts(
            f"{prompt_path}/{domain}.txt",
            iteration_limit=num_iterations,
            model_name=model_name,
            critique_model_name=critique_model_name,
        )

        # Save dataset
        dataset_path_parquet = output_path / f"ouroboros_{domain}_dataset.parquet"
        dataset_path_json = output_path / f"ouroboros_{domain}_dataset.json"

        dataset.to_parquet(str(dataset_path_parquet))  # Save as Parquet format
        dataset.to_json(str(dataset_path_json))  # Also save as JSON

        logging.info(
            f"Dataset successfully saved: {dataset_path_parquet}, {dataset_path_json}"
        )


if __name__ == "__main__":
    main()
