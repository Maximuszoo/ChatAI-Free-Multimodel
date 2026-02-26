"""Configuration manager for CAFM — loads, validates, and persists config.json."""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "instances": 3,
    "rounds": 3,
    "models": ["llama3.2", "qwen2.5", "mistral"],
    "context_limit": 4096,
    "context_strategy": "sliding_window",  # "sliding_window" | "summary"
    "summary_model": None,
    "stream_output": True,
    "save_logs": True,
    "log_directory": "logs",
    "skeptic_agent": True,
    "system_prompts": {
        "initial_round": (
            "You are an expert analyst participating in a multi-agent debate. "
            "Provide your independent, well-reasoned perspective on the user's query. "
            "Be thorough, specific, and support your arguments with clear reasoning."
        ),
        "debate_round": (
            "You are an expert analyst participating in a multi-agent debate. "
            "Review the full transcript of the debate so far. You MUST specifically "
            "refute, clarify, or expand on points raised by other participants. "
            "Reference their arguments directly. Do not simply repeat what has been "
            "said — add genuine value."
        ),
        "skeptic_round": (
            "You are the SKEPTIC in a multi-agent debate. Your sole role is to "
            "challenge, question, and actively refute the arguments made by the other "
            "participants in the transcript. Find logical flaws, missing evidence, "
            "hidden assumptions, and counter-examples. Be direct and adversarial. "
            "Do NOT simply agree — your job is to stress-test every claim."
        ),
        "skeptic_initial_round": (
            "You are the SKEPTIC in a multi-agent debate. Since this is the opening round "
            "and no other participant has spoken yet, your job is to challenge the USER'S "
            "QUERY ITSELF. Identify hidden assumptions, ambiguous language, missing context, "
            "or logical traps embedded in the question. Present an adversarial perspective "
            "that forces the debate to confront potential flaws from the very start. "
            "Do NOT give a straightforward answer — question whether the question is even "
            "well-formed."
        ),
        "final_synthesis": (
            "You are the final synthesizer in a multi-agent debate. Review the entire "
            "debate transcript carefully. Produce a single, comprehensive, high-quality "
            "answer that integrates the best insights from all participants. Resolve any "
            "contradictions and present the strongest possible consensus answer to the "
            "user's original query."
        ),
    },
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


class ConfigManager:
    """Manages loading, merging, and saving of CAFM configuration."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else CONFIG_PATH
        self._data: dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load config from disk, falling back to defaults for missing keys."""
        base = copy.deepcopy(DEFAULT_CONFIG)
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    user_cfg = json.load(fh)
                base = self._deep_merge(base, user_cfg)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[WARNING] Failed to read {self.path}: {exc}. Using defaults.")
        else:
            self.save(base)
        self._data = base
        return self._data

    def save(self, data: dict[str, Any] | None = None) -> None:
        """Persist current (or provided) config to disk."""
        payload = data if data is not None else self._data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def instances(self) -> int:
        return int(self._data.get("instances", 3))

    @property
    def rounds(self) -> int:
        return int(self._data.get("rounds", 3))

    @property
    def models(self) -> list[str]:
        return self._data.setdefault("models", [])

    def set_model_at(self, index: int, model_name: str) -> None:
        """Set a specific model by index and persist immediately."""
        models = self.models
        while len(models) <= index:
            models.append(models[-1] if models else "llama3.2")
        models[index] = model_name
        self.save()

    @property
    def context_limit(self) -> int:
        return int(self._data.get("context_limit", 4096))

    @property
    def context_strategy(self) -> str:
        return str(self._data.get("context_strategy", "sliding_window"))

    @property
    def skeptic_agent(self) -> bool:
        return bool(self._data.get("skeptic_agent", True))

    @property
    def system_prompts(self) -> dict[str, str]:
        return dict(self._data.get("system_prompts", DEFAULT_CONFIG["system_prompts"]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge *override* into *base*."""
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager._deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def ensure_models_match_instances(self) -> None:
        """Pad or trim the models list so it matches the instance count."""
        models = self.models
        n = self.instances
        if len(models) < n:
            filler = models[-1] if models else "llama3.2"
            models.extend([filler] * (n - len(models)))
        elif len(models) > n:
            models = models[:n]
        self._data["models"] = models
        self.save()
