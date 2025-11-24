# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class BdistWheel(bdist_wheel):
    """
    Custom wheel builder for a platform-specific Python package.

    - Marks the wheel as non-pure (`root_is_pure = False`) to ensure proper installation
      of native binaries.
    - Overrides the tag to be Python 3.11-specific (`cp311-cp311`) while preserving
      platform specificity.
    """

    user_options = bdist_wheel.user_options + [
        ("code-coverage", None, "Enable code coverage for the build")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.code_coverage = False  # Default value for code coverage

    def finalize_options(self):
        if self.code_coverage is None:
            self.code_coverage = False

        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        # Force specific Python 3.11 ABI format for the wheel
        python, abi = "cp311", "cp311"
        return python, abi, plat


setup(
    name="tt-vllm-plugin",
    cmdclass={
        "bdist_wheel": BdistWheel,
    },
    version="0.1.0",
    packages=["tt_vllm_plugin", "tt_vllm_plugin.worker", "tt_vllm_plugin.v1", 
              "tt_vllm_plugin.v1.worker", "tt_vllm_plugin.model_loader"],
    install_requires=[
        "vllm>=0.7.0",  # Flexible version to avoid conflicts
        "transformers>=4.40.0",
        # Note: ttnn should be installed separately as it's platform-specific
        # torch version should match vLLM's requirements
    ],
    python_requires=">=3.10, <3.12",
    license="Apache-2.0",
    entry_points={
        "vllm.platform_plugins": ["tt = tt_vllm_plugin:register"],
        "vllm.general_plugins": ["tt_model_registry = tt_vllm_plugin:register_models"],
    },
    description="vLLM plugin for Tenstorrent hardware acceleration",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)

