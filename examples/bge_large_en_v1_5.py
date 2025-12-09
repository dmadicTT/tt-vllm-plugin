#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Minimal example for running BGE-Large-EN-v1.5 embedding model on Tenstorrent hardware
using the TT vLLM plugin.

This example demonstrates basic embedding generation with the plugin.

Usage:
    python examples/bge_large_en_v1_5.py
"""

import os

# Enable vLLM v1 architecture
os.environ["VLLM_USE_V1"] = "1"
os.environ["HF_MODEL"] = "BAAI/bge-large-en-v1.5"

from vllm import LLM

def main():
    from vllm.platforms import current_platform  
    
    # Verify your platform is detected  
    assert current_platform.device_name == "tt", f"Expected 'tt' platform, got {current_platform.device_name}"  
    assert current_platform.is_out_of_tree(), "Platform should be OOT" 
    print(f"Using platform: {current_platform.device_name} (OOT: {current_platform.is_out_of_tree()})")
    
    print("Initializing BGE embedding model with TT platform...")
    print("Note: The TT plugin should be automatically discovered via entry points")
    
    llm = LLM(
        model='BAAI/bge-large-en-v1.5',
        max_model_len=384,
        max_num_seqs=32,
        enable_chunked_prefill=False,
        block_size=64,
        max_num_batched_tokens=2048,
        seed=9472,
    )
    
    print("Model loaded successfully!")
    
    # Test prompts for embedding generation
    prompts = [
        "What is the capital of France?",
        "Tell me about artificial intelligence.",
        "How does photosynthesis work?",
        "Explain quantum computing.",
        "What is the meaning of life?","What is the capital of France?",
        "Tell me about artificial intelligence.",
        
    ]
    
    print("\nGenerating embeddings...")
    outputs = llm.encode(prompts)
    
    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        # output is a PoolingRequestOutput object
        # Access the embedding via output.outputs.data (PoolingOutput has a 'data' attribute)
        embedding = output.outputs.data
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 10 dims): {embedding[:10]}")
    print("="*60)
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()




