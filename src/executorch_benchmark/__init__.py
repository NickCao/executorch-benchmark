import executorch.extension.pybindings._portable_lib
from executorch.extension.llm.runner import MultimodalRunner, GenerationConfig
from transformers import AutoProcessor
from PIL import Image
import torch


def main() -> None:
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    runner = MultimodalRunner(
        model_path="models/gemma-3-4b-it/model.pte",
        tokenizer_path="models/gemma-3-4b-it/tokenizer.json",
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, who is this?"},
            ],
        }
    ]

    # Apply chat template and process
    prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    inputs_hf = processor(images=None, text=prompt, return_tensors="pt")

    config = GenerationConfig(max_new_tokens=100, temperature=0.7)
    runner.generate_hf(
        inputs_hf,
        config,
        token_callback=lambda token: print(token, end="", flush=True),
    )
