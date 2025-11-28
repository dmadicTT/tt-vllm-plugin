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
os.environ["HF_MODEL"] = "/home/dmadic/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"

from vllm import LLM, SamplingParams, ModelRegistry


def main():
    from vllm.platforms import current_platform  
  
    # Verify your platform is detected  
    assert current_platform.device_name == "tt", f"Expected 'tt' platform, got {current_platform.device_name}"  
    assert current_platform.is_out_of_tree(), "Platform should be OOT" 
    print(f"Using platform: {current_platform.device_name} (OOT: {current_platform.is_out_of_tree()})")
    
    # Model configuration
    # model = "meta-llama/Llama-3.1-8B-Instruct"
    model = "/home/dmadic/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"

    print("Initializing LLM with TT platform...")
    print("Note: The TT plugin should be automatically discovered via entry points")
    
    llm = LLM(
        model=model,
        max_model_len=65536,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        block_size=64,
        max_num_batched_tokens=65536,
        seed=9472,
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

