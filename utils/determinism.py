from __future__ import annotations

import os


DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_deterministic_environment(
    *,
    cublas_workspace_config: str = DEFAULT_CUBLAS_WORKSPACE_CONFIG,
) -> dict[str, str]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cublas_workspace_config)
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
