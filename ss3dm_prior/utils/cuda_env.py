"""CUDA runtime environment helpers."""

from __future__ import annotations

import os
from pathlib import Path


def configure_isolated_mps_pipe_if_needed() -> str | None:
    """Avoid attaching to another user's global MPS daemon.

    Some shared machines expose a default `/tmp/nvidia-mps` created by another
    user. In that case, CUDA clients in this repo can hang during
    `torch.cuda.is_available()` or the first real allocation. When we detect
    that situation and no explicit MPS directory was requested, we point the
    current process at a private per-user pipe directory instead.
    """

    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        return None

    shared_pipe_dir = Path("/tmp/nvidia-mps")
    if not shared_pipe_dir.exists():
        return None

    try:
        stat_result = shared_pipe_dir.stat()
    except OSError:
        return None

    if stat_result.st_uid == os.getuid():
        return None

    control_socket = shared_pipe_dir / "control"
    if not control_socket.exists():
        return None

    username = os.environ.get("USER") or os.environ.get("USERNAME") or f"uid{os.getuid()}"
    private_pipe_dir = Path(f"/tmp/{username}-nvidia-mps")
    private_log_dir = Path(f"/tmp/{username}-nvidia-log")
    private_pipe_dir.mkdir(parents=True, exist_ok=True)
    private_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(private_pipe_dir)
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", str(private_log_dir))
    return str(private_pipe_dir)
