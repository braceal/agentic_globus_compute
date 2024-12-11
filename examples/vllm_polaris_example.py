"""VLLM example using the Polaris endpoint."""

from __future__ import annotations

import os

from agentic_globus_compute.vllm import run_vllm

if __name__ == '__main__':
    # Polaris endpoint ID
    os.environ['VLLM_ENDPOINT_ID'] = '1e4f9309-f8c0-45e4-aac6-f0cf39a87a42'

    # Sample prompts.
    prompts = [
        'Hello, my name is',
        'The president of the United States is',
        'The capital of France is',
        'The future of AI is',
    ]

    responses = run_vllm(
        prompts=prompts,
        pretrained_model_name_or_path='mosaicml/mpt-7b-storywriter',
    )

    for prompt, response in zip(prompts, responses):
        print(f'Prompt: {prompt}')
        print(f'Response: {response}')
        print()
