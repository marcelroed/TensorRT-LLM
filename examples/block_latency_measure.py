#!/usr/bin/env python3
"""Example demonstrating block prediction with TensorRT-LLM.

Block prediction allocates a block of N tokens, all starting as masked tokens,
then runs forward passes with no causal mask, unmasking tokens with softmax
probabilities greater than a threshold (e.g., 0.8), always unmasking at least one token.
This process repeats until all tokens are unmasked.
"""
from contextlib import contextmanager
import time


@contextmanager
def timer(name: str | None = None):
    """Context manager for timing code execution."""
    if name is None:
        name = "Execution"

    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed = end - start
        print(f"{name} took {elapsed:.2f} seconds")


with timer("Importing torch..."):
    import torch
with timer("Importing transformers..."):
    from transformers import AutoTokenizer
with timer("Importing tensorrt_llm..."):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig


def main():
    # Create an LLM with block prediction enabled
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",  # Using a supported model
        tokenizer=AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True),
        backend="pytorch",
        # enable_block_prediction=True,
        # disable_overlap_scheduler=True,
        # block_size=8,  # Number of tokens to predict in each block
        # keep_threshold=0.6,  # Confidence threshold for keeping tokens
        # mask_token_id=151666,  # Token ID to use as mask
        # max_iterations=10,  # Maximum number of iterations
        max_batch_size=1,
        max_num_tokens=131200,
        max_seq_len=131201,
        autotuner_enabled=True,
    )

    # Access block prediction config through the args
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    print()

    num_context_tokens_list = [
        128,
        1024,
        8192 - 32,
        # 131072,
    ]

    prompts = [
        'the ' * n for n in num_context_tokens_list
    ]

    # Define sample prompts
    # prompts = [
    #     # "The future of artificial intelligence is not the"+"<|mask|>"*8,
    #     # "In a world where technology advances rapidly,",
    #     # "The most important thing to remember is",
    #     "Come up with a way to compare strawberry jam and SIMD."
    # ]

    # Create sampling parameters
    # sampling_params = SamplingParams(
    #     max_tokens=1024,
    #     temperature=0.7,
    #     top_p=0.9,
    #     # end_id=151643,  # Add end_id to prevent ValueError
    # )
    # sampling_params = None
    sampling_params = SamplingParams(
        max_tokens=32,
    )

    # Generate text with block prediction
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}: {prompt}")

        # Generate response
        outputs = llm.generate([prompt], sampling_params)

        # Print the generated text
        # outputs is a list of RequestOutput objects
        # Each RequestOutput has an outputs attribute which is a list of CompletionOutput objects
        if isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
            if hasattr(output, 'outputs') and isinstance(output.outputs, list) and len(output.outputs) > 0:
                generated_text = output.outputs[0].text
                print(f"Generated: {generated_text}")
                print(f"  Generated {len(generated_text.split())} words")
            else:
                print("No output generated")
        else:
            print("No outputs returned")
        print()

if __name__ == "__main__":
    main()
