"""Test-suite environment. Imported by pytest before any test module.

The suite must never touch a real network. That is not automatic: ``app.py`` builds
its module-level ``ENGINE`` at import time, and ``GemmaLLM`` resolves its backend from
the *ambient* environment. A developer who has followed the README (``export
HF_TOKEN=...``, ``export GEMMA_BACKEND=hf_api``) would otherwise have ``import app``
alone send sample-inbox email bodies to the Hugging Face Inference API, and be billed
for it.

So the environment is pinned here, at module scope rather than in a fixture, because
conftest is imported before the test modules that do ``import app``. Individual tests
that need a different environment (``tests/test_backend.py``) override it with
``monkeypatch``, which restores these values afterwards.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The one backend that needs no weights, no token and no network.
os.environ["GEMMA_BACKEND"] = "heuristic"
os.environ["ALLOW_REMOTE_INFERENCE"] = "false"

# Any credential the ambient shell is carrying is not this suite's to spend.
for _name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MAX_ACTIONS_PER_EMAIL"):
    os.environ.pop(_name, None)

# Belt and braces: even a code path that ignored the settings above cannot reach the
# Hub, and neither Gradio nor huggingface_hub may phone home with usage telemetry.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("DO_NOT_TRACK", "1")
