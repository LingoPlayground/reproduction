"""Pipeline environment loading — call once from a CLI entrypoint, never at module import."""
from __future__ import annotations

import os
from pathlib import Path


def load_pipeline_env() -> None:
    """Load environment variables from downstream project .env files.
    
    Only called from CLI entrypoints (main functions), never at module import time.
    Uses setdefault so explicit env vars take priority over .env files.
    """
    env_paths = [
        Path("~/workspace/lingolens/backend/.env").expanduser(),
        Path("~/workspace/shakespeare/.env").expanduser(),
    ]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
