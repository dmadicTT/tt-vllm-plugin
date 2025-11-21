# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Basic offline inference test for TT vLLM plugin.

This test verifies that the plugin can perform basic offline inference
on Tenstorrent hardware. It uses the meta-llama/Llama-3.1-8B-Instruct model
as a minimal example.

To run this test:
1. Ensure you have Tenstorrent hardware available
2. Set VLLM_USE_V1=1 environment variable
3. Install the plugin: pip install -e .
4. Run: python -m pytest tt_vllm_plugin/tests/test_offline_inference.py -v
"""

import os
import pytest

# Skip if TT hardware is not available
try:
    import ttnn

    device_ids = ttnn.get_device_ids()
    if len(device_ids) == 0:
        pytest.skip("No TT devices available", allow_module_level=True)
except (ImportError, RuntimeError):
    pytest.skip(
        "ttnn not available or TT hardware not accessible", allow_module_level=True
    )


def test_offline_inference_basic():
    """Test basic offline inference with Llama-3.1-8B-Instruct."""
    import os

    os.environ["VLLM_USE_V1"] = "1"

    from vllm import LLM, SamplingParams

    # Model configuration
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Initialize LLM with TT platform
    # The plugin should be automatically discovered via entry points
    llm = LLM(
        model=model,
        device="tt",  # Use TT platform
        max_model_len=2048,
        max_num_seqs=1,
        # TT-specific configuration
        override_tt_config={
            "trace_mode": True,
        },
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=10,
    )

    # Test prompt
    prompts = ["Hello, my name is"]

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Verify output
    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].token_ids) > 0

    print(f"Generated text: {outputs[0].outputs[0].text}")
    print("Test passed: Basic offline inference works!")


if __name__ == "__main__":
    test_offline_inference_basic()
