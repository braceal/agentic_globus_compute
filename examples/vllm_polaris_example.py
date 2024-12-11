"""VLLM example using the Polaris endpoint."""

from __future__ import annotations

import os
import time

if __name__ == '__main__':
    # Time the execution
    start = time.time()

    # Polaris endpoint ID (run this before the import)
    os.environ['VLLM_ENDPOINT_ID'] = '1e4f9309-f8c0-45e4-aac6-f0cf39a87a42'

    from agentic_globus_compute.vllm import run_vllm

    # Sample prompts
    prompts = [
        'Hello, my name is',
        'The president of the United States is',
        'The capital of France is',
        'The future of AI is',
    ]

    # Generate responses from the prompts
    responses = run_vllm(
        prompts=prompts,
        pretrained_model_name_or_path='allenai/OLMo-1B-hf',
    )

    # Print the responses
    print(f'Time taken: {time.time() - start:.2f} seconds')
    for prompt, response in zip(prompts, responses):
        print(f'Prompt: {prompt}')
        print(f'Response: {response}')
        print()
