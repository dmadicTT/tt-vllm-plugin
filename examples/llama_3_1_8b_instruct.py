#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Minimal example for running Llama-3.1-8B-Instruct on Tenstorrent hardware
using the TT vLLM plugin.

This example demonstrates basic offline inference with the plugin.

Usage:
    export VLLM_USE_V1=1
    python examples/llama_3_1_8b_instruct.py
"""

import os

# Enable vLLM v1 architecture
os.environ["VLLM_USE_V1"] = "1"

from vllm import LLM, SamplingParams


def main():
    # Model configuration
    model = "meta-llama/Llama-3.1-8B-Instruct"
    
    print("Initializing LLM with TT platform...")
    print("Note: The TT plugin should be automatically discovered via entry points")
    
    # Initialize LLM with TT platform
    llm = LLM(
        model=model,
        device="tt",  # Use TT platform (plugin will be loaded automatically)
        max_model_len=2048,
        max_num_seqs=1,
        # TT-specific configuration
        override_tt_config={
            "trace_mode": True,
        }
    )
    
    print("Model loaded successfully!")
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=50,
    )
    
    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    
    print("\nGenerating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    for i, output in enumerate(outputs):
        prompt = prompts[i]
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Token IDs: {output.outputs[0].token_ids[:10]}...")  # First 10 tokens
    print("="*60)
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()

