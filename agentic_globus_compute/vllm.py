"""VLLMGenerator class for generating text using the VLLM backend."""

from __future__ import annotations

import os

from agentic_globus_compute import globus_compute_executor


class VLLMGenerator:
    """Language model generator using vllm backend.

    See here for more details: https://github.com/vllm-project/vllm
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        temperature: float = 0.0,
        min_p: float = 0.1,
        top_p: float = 0.0,
        max_tokens: int = 500,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_seq_len_to_capture: int = 45000,
        enforce_eager: bool = True,
    ) -> None:
        """Initialize the VLLMGenerator.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Pretrained model name or path to use.
        trust_remote_code : bool, optional
            Whether to trust remote code, by default True
        temperature : float, optional
            Temperature for sampling, by default 0.0
        min_p : float, optional
            Min p for sampling, by default 0.1
        top_p : float, optional
            Top p for sampling (off by default), by default 0.0
        max_tokens : int, optional
            Max tokens to generate, by default 500
        tensor_parallel_size : int, optional
            The number of GPUs to use, by default 1
        gpu_memory_utilization : float, optional
            percentage of gpu to utilize, by default 0.9
        max_seq_len_to_capture : int, optional
            The maximum sequence length to capture, by default 45000
        enforce_eager : bool, optional
            no cuda graph, by default True
        """
        from vllm import LLM
        from vllm import SamplingParams

        # Create the sampling params to use
        sampling_kwargs = {}
        if top_p:
            sampling_kwargs['top_p'] = top_p
        else:
            sampling_kwargs['min_p'] = min_p

        # TODO: Support beam search?

        # Create the sampling params to use
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **sampling_kwargs,
        )

        # Create an LLM instance
        model = LLM(
            model=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype='bfloat16',
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_seq_len_to_capture=max_seq_len_to_capture,
            enforce_eager=enforce_eager,
        )

        self.model = model

    def generate(self, prompts: str | list[str]) -> list[str]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : str | list[str]
            The prompts to generate text from.

        Returns
        -------
        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Generate responses from the prompts. The output is a list of
        # RequestOutput objects that contain the prompt, generated text,
        # and other information.
        outputs = self.model.generate(prompts, self.sampling_params)

        # Extract the response from the outputs
        responses = [output.outputs[0].text for output in outputs]

        return responses


@globus_compute_executor(endpoint_id=os.environ.get('VLLM_ENDPOINT_ID'))
def run_vllm(
    prompts: str | list[str],
    pretrained_model_name_or_path: str,
    trust_remote_code: bool = True,
    temperature: float = 0.0,
    min_p: float = 0.1,
    top_p: float = 0.0,
    max_tokens: int = 500,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_seq_len_to_capture: int = 45000,
    enforce_eager: bool = True,
) -> list[str]:
    """Run the VLLM model to generate text from prompts.

    Parameters
    ----------
    prompts : str | list[str]
        The prompts to generate text from.
    pretrained_model_name_or_path : str
        Pretrained model name or path to use.
    trust_remote_code : bool, optional
        Whether to trust remote code, by default True
    temperature : float, optional
        Temperature for sampling, by default 0.0
    min_p : float, optional
        Min p for sampling, by default 0.1
    top_p : float, optional
        Top p for sampling (off by default), by default 0.0
    max_tokens : int, optional
        Max tokens to generate, by default 500
    tensor_parallel_size : int, optional
        The number of GPUs to use, by default 1
    gpu_memory_utilization : float, optional
        percentage of gpu to utilize, by default 0.9
    max_seq_len_to_capture : int, optional
        The maximum sequence length to capture, by default 45000
    enforce_eager : bool, optional
        no cuda graph, by default True

    Returns
    -------
    list[str]
        A list of responses generated from the prompts
        (one response per prompt).
    """
    from parsl_object_registry import registry

    from agentic_globus_compute.vllm import VLLMGenerator

    # Register the VLLMGenerator for warm start on subsequent calls
    registry.register(VLLMGenerator)

    # Initialize the VLLMGenerator
    llm = registry.get(
        VLLMGenerator,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        temperature=temperature,
        min_p=min_p,
        top_p=top_p,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_seq_len_to_capture=max_seq_len_to_capture,
        enforce_eager=enforce_eager,
    )

    # Generate the text
    responses = llm.generate(prompts)

    return responses
