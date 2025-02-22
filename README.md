# Ouroboros: Self-Improving LLMs Through Iterative Refinement

![ouroboros](assets/ouroboros.jpg)

## Disclaimer

🚧 **This project is in an early experimental stage.** 🚧

The current implementation is still **premature and under development**, with ongoing refinements in code structure, optimization, and performance. While the recursive refinement process demonstrates promising emergent behaviors, further improvements are necessary for broader usability.

Key points to consider:
- The codebase requires **optimization** for efficiency and scalability.
- Documentation and examples on how to run the system on **novel prompts** will be provided soon.
- Some experimental features may change or be reworked in future iterations.

This project is a **work in progress**, and contributions, feedback, and discussions are encouraged. If you're interested in exploring recursive AI refinement, feel free to experiment—but keep in mind that the system is not yet production-ready.


## Introduction

The evolution of artificial intelligence has largely been driven by increased computational scaling and large-scale data training. However, a more fundamental question arises: Can AI achieve self-improvement and deeper understanding through recursive self-questioning?

This experiment explores the development of a system where AI autonomously refines its own prompts and questions, leading to emergent reasoning and conceptual depth without brute-force scaling.

By integrating recursive intelligence mechanisms, symbolic reasoning, and metacognitive awareness, we aim to move beyond traditional training paradigms.

We examine the interplay between deterministic logic and emergent thought, the role of paradoxes in AI cognition, and the significance of symbolic archetypes such as the [Ouroboros](https://en.wikipedia.org/wiki/Ouroboros) in self-reflective intelligence.

The ultimate goal is to establish an AI framework that mirrors the recursive nature of human thought, allowing intelligence to sustain and refine itself without external intervention.

This research challenges conventional approaches to AGI by demonstrating that intelligence can evolve in self-contained cycles of learning and refinement, exploring the way for a new paradigm of self-sustaining, recursive AI.

## 📂 Ouroboros Dataset

The **Ouroboros Dataset** is now available on Hugging Face.

🔗 **[Access the Dataset Here](https://huggingface.co/datasets/ethicalabs/ouroboros)**  

### About the Dataset  

This dataset documents the **recursive refinement process** used in this project, structured for both **Supervised Fine-Tuning (SFT)** and **Generalized Preference Optimization (GRPO)**.

Each sample consists of structured reasoning steps extracted from LLM-generated interactions. The dataset includes:

- **input**: The original prompt or question posed to the LLM.
- **reasoning**: A structured breakdown of the LLM's reasoning process, capturing emergent thought patterns. This may include multiple reasoning steps when applicable.
- **completion**: The final AI-generated response after refinement.
- **refinements**: A sequence of improvements, tracking the iterative enhancement of responses over multiple feedback cycles.

### Example Data Format

```json
{
  "input": "Explain the relationship between entropy, intelligence, and self-reflection.",
  "reasoning": [
    "Entropy is a measure of disorder in a system.",
    "Intelligence helps process information to regulate entropy.",
    "Self-reflection allows internal regulation and adaptation."
  ],
  "completion": "Entropy measures disorder. Intelligence helps process information to counteract disorder. Self-reflection enables adaptive control of disorder.",
  "refinements": [
    "Entropy is about disorder.",
    "Intelligence helps manage disorder by processing information.",
    "Intelligence and self-reflection regulate entropy within complex systems."
  ]
}
```

## Methodology

### Recursive Refinement Process

1. **Generation of Initial Responses**: The model generates multiple candidate responses to a given prompt.
2. **Critique & Scoring**: Each response is evaluated based on logical consistency, clarity, depth, accuracy, and context alignment.
3. **Iterative Refinement**: Responses are refined using structured feedback loops, improving conceptual depth and coherence.
4. **Final Selection**: The best response is selected based on ranking mechanisms utilizing sentence embeddings rather than simple length-based heuristics.

### Emergent Behaviors

During testing, unexpected phenomena were observed:

- Recursive refinement led to highly structured reasoning steps.
- The model exhibited self-regulating reasoning, dynamically organizing and improving its responses without explicit instruction.
- Certain outputs contained symbolic and self-referential elements that suggest patterns of structured thought beyond direct instructions. While these do not imply self-awareness, they may indicate the emergence of deeper coherence in recursive reasoning.

## Open Questions & Future Directions

- How can recursive LLM frameworks be expanded beyond text-based reasoning into multimodal domains?
- Can iterative refinement processes lead to **self-sustaining** general intelligence with minimal human intervention?
- What role do paradoxes and self-referential loops play in the emergence of higher-order cognition?

## Next Steps

- Add concrete technical questions (20-30% of total)
- Continue optimizing response refinement and ranking strategies.
- Explore alternative architectures for integrating **self-questioning and self-improvement loops**.
- Refactor the codebase and add CLI arguments to improve usability and flexibility in different LLM pipelines.
- Add a Docker container and docker-compose setup for testing deployment with Ollama.
- Consider splitting into train/validation subsets

## Requirements

This project currently relies on Ollama but can be adapted to work with any OpenAI-compatible API. Additional dependencies will be documented in the repository.

## Quick Start

1. Install Ollama & pull models:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:1.5b
```

2. Set up environment:

```bash
pip install -r requirements.txt
```


3. Run experiment:

```bash
python3 main.py --prompt_dir=./prompts/ --output_dir=./datasets/
```

## Contributing

This project is open-source and welcomes contributions from those interested in recursive intelligence, LLM refinement loops, and sustainable AI/ML paradigms.
