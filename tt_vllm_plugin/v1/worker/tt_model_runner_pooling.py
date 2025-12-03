# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Pooling Model Runner for TT Platform

This module implements a simplified model runner for pooling/embedding models.
Unlike generation models, pooling models:
- Don't require KV cache
- Don't require page tables
- Don't have decode steps
- Simply take tokens and return embeddings
"""

import os
from typing import TYPE_CHECKING, Optional

import torch
import ttnn
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.outputs import ModelRunnerOutput
from vllm.tasks import PoolingTask, SupportedTask

from tt_vllm_plugin.model_loader.tt_loader import TTModelLoader

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class TTModelRunnerPooling:
    """
    Simplified model runner for pooling/embedding models.
    
    This runner is much simpler than TTModelRunner because pooling models don't need:
    - KV cache management
    - Page tables
    - Decode steps
    - Prefill/decode split
    - Sampling logic
    
    They simply take tokenized inputs and return embeddings.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: bool,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config

        self.mesh_device = mesh_device
        self.trace_mode = trace_mode

        logger.info(
            "TTModelRunnerPooling: trace_mode=%s",
            self.trace_mode,
        )

        # Track request states (simpler than generation models)
        self.requests: dict[str, dict] = {}
        
        # Model will be set by load_model() or directly by worker
        self.model: Optional[nn.Module] = None

    def load_model(self) -> None:
        """Load the pooling model."""
        if self.model is not None:
            # Model already loaded (e.g., by worker)
            logger.info("Pooling model already loaded, skipping")
            return
        logger.info("Loading TT pooling model...")
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(
            vllm_config=self.vllm_config,
            model_config=self.model_config
        )

    def _prepare_model_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare model inputs for pooling models.
        
        Args:
            scheduler_output: Scheduler output with scheduled requests
            
        Returns:
            Tuple of (tokens, attention_mask) tensors
        """
        # Gather all scheduled requests
        scheduled_reqs = scheduler_output.scheduled_new_reqs
        
        if not scheduled_reqs:
            return None, None
        
        # Get token IDs and create attention masks
        token_ids_list = []
        attention_mask_list = []
        max_seq_len = 0
        
        for req_data in scheduled_reqs:
            req_id = req_data.req_id
            prompt_token_ids = req_data.prompt_token_ids
            
            # Store request info
            self.requests[req_id] = {
                "prompt_token_ids": prompt_token_ids,
                "pooling_params": req_data.pooling_params,
            }
            
            seq_len = len(prompt_token_ids)
            max_seq_len = max(max_seq_len, seq_len)
            token_ids_list.append(prompt_token_ids)
        
        # Pad all sequences to max_seq_len
        batch_size = len(token_ids_list)
        tokens = torch.zeros((batch_size, max_seq_len), dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
        
        for i, token_ids in enumerate(token_ids_list):
            seq_len = len(token_ids)
            tokens[i, :seq_len] = torch.tensor(token_ids, dtype=torch.int64)
            attention_mask[i, :seq_len] = 1.0
        
        return tokens, attention_mask

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        """
        Execute the pooling model with the given scheduler output.
        
        Args:
            scheduler_output: Scheduler output with scheduled requests
            
        Returns:
            ModelRunnerOutput with embeddings in pooler_output field
        """
        # Prepare inputs
        tokens, attention_mask = self._prepare_model_inputs(scheduler_output)
        
        if tokens is None:
            # No requests to process
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
        
        # Execute model forward pass
        assert self.model is not None, "Model not loaded. Call load_model() first."
        embeddings = self.model.forward(
            input_ids=tokens,
            attention_mask=attention_mask,
        )
        
        # Convert embeddings to list of tensors format expected by vLLM
        # embeddings shape: [batch_size, embedding_dim]
        # pooler_output should be list[Optional[torch.Tensor]] - one tensor per request
        batch_size = embeddings.shape[0]
        pooler_output = [embeddings[i].cpu() for i in range(batch_size)]
        
        # Get request IDs in order
        req_ids = [req_data.req_id for req_data in scheduler_output.scheduled_new_reqs]
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
        
        # sampled_token_ids should be list[list[int]] - one list per request (empty for pooling)
        sampled_token_ids = [[] for _ in range(batch_size)]
        
        # Clean up finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,  # List of empty lists for pooling models
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    def get_model(self) -> nn.Module:
        """Get the underlying model."""
        assert self.model is not None, "Model not loaded. Call load_model() first."
        return self.model

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """Get supported pooling tasks for this model."""
        model = self.get_model()
        
        # Check if model has get_embedding_dim or similar methods
        # For now, assume all pooling models support "embed" task
        if hasattr(model, "get_embedding_dim"):
            return ["embed"]
        
        # Fallback: check if model has forward method that returns embeddings
        # This is a simple heuristic - in practice, models should implement
        # the proper interface
        return ["embed"]

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """Get all supported tasks for this model runner."""
        return tuple(self.get_supported_pooling_tasks())

