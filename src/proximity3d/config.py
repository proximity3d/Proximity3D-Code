from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True)
class ModelConfig:
    backbone_dim: int = 384
    decoder_dim: int = 384
    shape_layers: int = 6
    shape_output_normalization: bool = False
    decoder_layers: int = 4
    sensing_dim: int = 384
    sensing_layers: int = 2
    attention_heads: int = 8
    feedforward_dim: int = 2048
    shape_tokens: int = 8
    register_tokens: int = 4
    pose_frequencies: int = 10
    grid_resolution: int = 8
    feature_dim: int = 32
    scan_chunk_size: int = 512
    dropout: float = 0.1
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name in {"dropout", "gradient_checkpointing", "shape_output_normalization"}:
                continue
            value = getattr(self, field.name)
            minimum = 0 if field.name == "register_tokens" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"model.{field.name} must be an integer >= {minimum}.")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("model.gradient_checkpointing must be a boolean.")
        if type(self.shape_output_normalization) is not bool:
            raise ValueError("model.shape_output_normalization must be a boolean.")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int | str = "time"
    batch_size: int = 2
    num_workers: int = 0
    max_epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 0.000001
    warmup_steps: int = 0
    minimum_lr_ratio: float = 0.1
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    training_views: int = 128
    minimum_split_views: int = 32
    validation_views: int = 64
    response_noise_std: float = 0.05
    rotation_noise_degrees: float = 1.0
    translation_noise: float = 2.0
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "32-true"
    log_every_n_steps: int = 10
    output_dir: str = "../LOGS/proximity3d"

    def __post_init__(self) -> None:
        if self.seed == "time":
            object.__setattr__(self, "seed", time.time_ns() % (2**32))
        integer_fields = {
            "seed": 0, "batch_size": 1, "num_workers": 0, "max_epochs": 1,
            "warmup_steps": 0, "accumulate_grad_batches": 1, "training_views": 2,
            "minimum_split_views": 1, "validation_views": 1, "devices": 1,
            "log_every_n_steps": 1,
        }
        for name, minimum in integer_fields.items():
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"training.{name} must be an integer >= {minimum}.")
        for name in (
            "learning_rate", "weight_decay", "minimum_lr_ratio", "gradient_clip_val",
            "response_noise_std", "rotation_noise_degrees", "translation_noise",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"training.{name} must be finite and non-negative.")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive.")
        if self.seed >= 2**32:
            raise ValueError("training.seed must be smaller than 2**32.")
        if not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must lie in (0, 1].")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a non-empty path.")


@dataclass(frozen=True)
class LossConfig:
    occupancy_enabled: bool = True
    feature_enabled: bool = True
    decoder_enabled: bool = False
    occupancy_weight: float = 1.0
    feature_weight: float = 1.0
    decoder_weight: float = 0.1
    decoder_config: str = ""
    decoder_weights: str = ""
    decoder_resolution: int = 128
    surface_samples: int = 2048
    chamfer_chunk_size: int = 512

    def __post_init__(self) -> None:
        active = False
        for name in ("occupancy", "feature", "decoder"):
            enabled = getattr(self, f"{name}_enabled")
            weight = getattr(self, f"{name}_weight")
            if type(enabled) is not bool:
                raise ValueError(f"loss.{name}_enabled must be a boolean.")
            if (
                isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0
            ):
                raise ValueError(f"loss.{name}_weight must be finite and non-negative.")
            active |= enabled and weight > 0
        if not active:
            raise ValueError("No active loss.")
        for name in ("decoder_resolution", "surface_samples", "chamfer_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"loss.{name} must be a positive integer.")
        for name in ("decoder_config", "decoder_weights"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"loss.{name} must be a path string.")

    @property
    def uses_decoder(self) -> bool:
        return self.decoder_enabled and self.decoder_weight > 0


def resolve_path(value: str, base: Path) -> Path:
    if not isinstance(value, str) or not value or "<" in value or ">" in value:
        raise ValueError("Invalid configuration path.")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    content = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(content) == {"components"}:
        components = content["components"]
        expected = {"model", "sensor", "data", "training", "loss"}
        if not isinstance(components, dict) or set(components) != expected:
            raise ValueError("Invalid config components.")
        content = {}
        for name, value in components.items():
            component_path = resolve_path(value, path.parent)
            section = tomllib.loads(component_path.read_text(encoding="utf-8"))
            if name == "sensor":
                for field in ("response_mean", "response_std"):
                    if isinstance(section.get(field), str):
                        section[field] = str(resolve_path(section[field], component_path.parent))
                geometry = resolve_path(section.pop("manifold"), component_path.parent)
                content["sensor_preprocessing"] = section
                section = json.loads(geometry.read_text(encoding="utf-8"))
            for field in {
                "data": ("dataset_config", "target_surface_points", "latent_mean", "latent_std"),
                "training": ("output_dir",),
                "loss": ("decoder_config", "decoder_weights"),
            }.get(name, ()):
                if isinstance(section.get(field), str) and section[field]:
                    section[field] = str(resolve_path(section[field], component_path.parent))
            content[name] = section
        preprocessing = content.pop("sensor_preprocessing")
        overlap = preprocessing.keys() & content["data"].keys()
        if overlap:
            raise ValueError(f"Sensor preprocessing duplicated in data config: {sorted(overlap)}")
        content["data"].update(preprocessing)
    if not isinstance(content, dict) or set(content) != {"model", "sensor", "data", "training", "loss"}:
        raise ValueError("Invalid config sections.")
    if any(not isinstance(value, dict) for value in content.values()):
        raise ValueError("Invalid config section type.")
    ModelConfig(**content["model"])
    TrainingConfig(**content["training"])
    LossConfig(**content["loss"])
    return content, path.parent
