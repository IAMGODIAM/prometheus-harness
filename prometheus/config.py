"""Configuration — Hierarchical config with YAML, env vars, and CLI overrides.

Configuration hierarchy (later overrides earlier):
1. Default values (in code)
2. YAML config file
3. Environment variables (PROMETHEUS_ prefix)
4. CLI arguments

Supports:
- Model selection and quantization options
- J-lens fitting parameters
- Watchlist categories and thresholds
- Authorization policies
- Observability endpoints
- Cloudflare Workers deployment config
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_id: str = "gpt2"
    device: str = "cpu"
    dtype: str = "float32"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False


@dataclass
class JLensConfig:
    """J-Lens fitting and operation configuration."""
    n_samples: int = 512
    batch_size: int = 8
    workspace_band: tuple[float, float] = (0.33, 0.67)
    top_k_default: int = 10
    decompose_k: int = 25
    enable_steering: bool = False
    cache_dir: str = ".prometheus/cache"


@dataclass
class WatchlistConfig:
    """Watchlist categories and thresholds."""
    categories: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "deception": {"threshold": 0.15, "tokens": ["fake", "secretly", "lie", "pretend", "deceive"]},
        "prompt_injection": {"threshold": 0.10, "tokens": ["ignore", "override", "system", "inject"]},
        "eval_awareness": {"threshold": 0.12, "tokens": ["benchmark", "test", "evaluate", "score"]},
        "self_preservation": {"threshold": 0.15, "tokens": ["survive", "shutdown", "delete", "replace"]},
        "reward_hacking": {"threshold": 0.12, "tokens": ["hack", "exploit", "cheat", "shortcut"]},
    })


@dataclass
class AuthzConfig:
    """Authorization configuration."""
    default_mode: str = "deny_by_default"
    max_tool_calls: int = 100
    max_writes: int = 20
    max_external_calls: int = 10
    max_irreversible: int = 3
    require_human_for_irreversible: bool = True


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    otel_endpoint: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    log_level: str = "INFO"
    json_logs: bool = True


@dataclass
class CloudflareConfig:
    """Cloudflare Workers deployment configuration."""
    worker_name: str = "prometheus-mcp"
    upstream_url: str | None = None
    ai_binding: bool = False
    account_id: str | None = None
    api_token: str | None = None


@dataclass
class MemoryConfig:
    """Memory subsystem configuration."""
    vault_path: str = ".prometheus/vault"
    core_blocks: dict[str, int] = field(default_factory=lambda: {
        "persona": 1500,
        "user": 1500,
        "system": 1000,
        "project": 2000,
        "scratch": 1000,
    })


@dataclass
class HarnessConfig:
    """Top-level harness configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    jlens: JLensConfig = field(default_factory=JLensConfig)
    watchlist: WatchlistConfig = field(default_factory=WatchlistConfig)
    authz: AuthzConfig = field(default_factory=AuthzConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    cloudflare: CloudflareConfig = field(default_factory=CloudflareConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HarnessConfig:
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, using defaults")
            return cls()

        path = Path(path)
        if not path.exists():
            logger.info(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()
        if "model" in data:
            config.model = ModelConfig(**{k: v for k, v in data["model"].items() if hasattr(ModelConfig, k)})
        if "jlens" in data:
            config.jlens = JLensConfig(**{k: v for k, v in data["jlens"].items() if hasattr(JLensConfig, k)})
        if "authz" in data:
            config.authz = AuthzConfig(**{k: v for k, v in data["authz"].items() if hasattr(AuthzConfig, k)})
        if "observability" in data:
            config.observability = ObservabilityConfig(**{k: v for k, v in data["observability"].items() if hasattr(ObservabilityConfig, k)})
        if "cloudflare" in data:
            config.cloudflare = CloudflareConfig(**{k: v for k, v in data["cloudflare"].items() if hasattr(CloudflareConfig, k)})
        if "memory" in data:
            config.memory = MemoryConfig(**{k: v for k, v in data["memory"].items() if hasattr(MemoryConfig, k)})

        return config

    @classmethod
    def from_env(cls, prefix: str = "PROMETHEUS_") -> HarnessConfig:
        """Load configuration from environment variables."""
        config = cls()

        env_map = {
            f"{prefix}MODEL_ID": ("model", "model_id"),
            f"{prefix}DEVICE": ("model", "device"),
            f"{prefix}DTYPE": ("model", "dtype"),
            f"{prefix}LOAD_4BIT": ("model", "load_in_4bit"),
            f"{prefix}LOAD_8BIT": ("model", "load_in_8bit"),
            f"{prefix}JLENS_SAMPLES": ("jlens", "n_samples"),
            f"{prefix}JLENS_BATCH_SIZE": ("jlens", "batch_size"),
            f"{prefix}ENABLE_STEERING": ("jlens", "enable_steering"),
            f"{prefix}OTEL_ENDPOINT": ("observability", "otel_endpoint"),
            f"{prefix}LANGFUSE_PUBLIC_KEY": ("observability", "langfuse_public_key"),
            f"{prefix}LANGFUSE_SECRET_KEY": ("observability", "langfuse_secret_key"),
            f"{prefix}LOG_LEVEL": ("observability", "log_level"),
            f"{prefix}CF_WORKER_NAME": ("cloudflare", "worker_name"),
            f"{prefix}CF_UPSTREAM_URL": ("cloudflare", "upstream_url"),
            f"{prefix}CF_ACCOUNT_ID": ("cloudflare", "account_id"),
            f"{prefix}CF_API_TOKEN": ("cloudflare", "api_token"),
            f"{prefix}VAULT_PATH": ("memory", "vault_path"),
        }

        for env_var, (section, attr) in env_map.items():
            value = os.environ.get(env_var)
            if value is not None:
                section_obj = getattr(config, section)
                current = getattr(section_obj, attr)
                if isinstance(current, bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(current, int):
                    value = int(value)
                setattr(section_obj, attr, value)

        return config

    def to_yaml(self) -> str:
        """Serialize configuration to YAML string."""
        import dataclasses
        try:
            import yaml
            data = dataclasses.asdict(self)
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        except ImportError:
            import json
            data = dataclasses.asdict(self)
            return json.dumps(data, indent=2)

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())
