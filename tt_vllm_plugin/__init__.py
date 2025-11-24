# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

def register():
    # At first we used ttnn.get_device_ids() to truly understand if the TT platform is supported.
    # This caused the offline inference to hang and never complete, so for now we just assume that we always have TT support.
    return "tt_vllm_plugin.platform.TTPlatform"


def register_models():
    """Register custom models with ModelRegistry for online inference.
    
    This function is called automatically by vLLM when the plugin is loaded,
    ensuring models are registered before the API server or engine starts.
    """
    from vllm import ModelRegistry
    
    # Register TT Llama model
    ModelRegistry.register_model(
        "TTLlamaForCausalLM", 
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    )
    
    # Add additional model registrations here as needed
    # ModelRegistry.register_model("AnotherModel", "path.to:ModelClass")