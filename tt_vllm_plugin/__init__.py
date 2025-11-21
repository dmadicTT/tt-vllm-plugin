# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os


def register():
    """Register the TT platform plugin.
    
    Returns the fully qualified name of the TTPlatform class if TT hardware
    is available, otherwise returns None.
    """
    # Setting worker multiprocessing method to spawn to avoid hangs in consecutive vllm pytest runs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Check if TT hardware is available
    try:
        import ttnn
        # Try to get device IDs to verify TT hardware is available
        device_ids = ttnn.get_device_ids()
        if len(device_ids) > 0:
            return "tt_vllm_plugin.platform.TTPlatform"
    except (ImportError, RuntimeError, Exception):
        # TT hardware not available or ttnn not installed
        pass
    
    return None

