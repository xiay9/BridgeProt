from __future__ import annotations

from dataclasses import dataclass
import os
import random
import subprocess

from .determinism import configure_deterministic_environment


configure_deterministic_environment()

import numpy as np
import torch


@dataclass(slots=True)
class RuntimeContext:
    device: torch.device
    is_main_process: bool


def setup_runtime(
    *,
    require_cuda: bool = False,
    initialize_cuda: bool = True,
) -> RuntimeContext:
    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise RuntimeError("CUDA is required for this script, but no GPU is available.")

    if has_cuda:
        if initialize_cuda:
            torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return RuntimeContext(
        device=device,
        is_main_process=True,
    )


def has_visible_cuda_gpu() -> bool:
    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw_visible_devices in {"-1", "none", "None"}:
        return False
    if raw_visible_devices:
        return any(token.strip() for token in raw_visible_devices.split(","))

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return any(line.strip().startswith("GPU ") for line in result.stdout.splitlines())


def cleanup_runtime() -> None:
    return None


def enable_deterministic_torch(*, warn_only: bool = False) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def set_random_seed(
    seed: int,
    *,
    deterministic: bool = False,
    deterministic_warn_only: bool = False,
    include_cuda: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        enable_deterministic_torch(warn_only=deterministic_warn_only)
