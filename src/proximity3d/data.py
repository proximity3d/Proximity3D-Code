from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from proximity3d.config import LossConfig, ModelConfig, TrainingConfig, resolve_path
from proximity3d.geometry import relative_postures
from proximity3d.dataset_config import discover_samples, split_samples


def load_tensor_file(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def require_tensor(value: Any, name: str, shape: tuple[int | None, ...]) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != len(shape):
        raise ValueError(f"{name} must be a tensor with shape {shape}.")
    if any(size is not None and actual != size for actual, size in zip(value.shape, shape)):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values.")
    return value


def normalization_vector(value: Any, size: int, base: Path, name: str, positive: bool = False) -> Tensor:
    if isinstance(value, str):
        value = np.load(resolve_path(value, base), allow_pickle=False)
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() not in {1, size} or not torch.isfinite(tensor).all():
        raise ValueError(f"data.{name} must contain 1 or {size} finite values.")
    if positive and (tensor <= 0).any():
        raise ValueError(f"data.{name} must be positive.")
    return tensor


def sparse_to_dense_target(coords: Tensor, feats: Tensor, config: ModelConfig, mean: Tensor, std: Tensor) -> Tensor:
    coords = require_tensor(coords, "coords", (None, 4))
    feats = require_tensor(feats, "feats", (len(coords), config.feature_dim)).float()
    if coords.dtype not in {torch.int16, torch.int32, torch.int64, torch.uint8, torch.int8}:
        raise ValueError("coords must contain integer coordinates.")
    if len(coords) == 0 or (coords[:, 0] != 0).any():
        raise ValueError("Empty or invalid latent coordinates.")
    xyz = coords[:, 1:].long()
    if (xyz < 0).any() or (xyz >= config.grid_resolution).any():
        raise ValueError("Latent coordinates are outside the configured grid.")
    if len(torch.unique(xyz, dim=0)) != len(xyz):
        raise ValueError("Latent coordinates must be unique.")
    dense = torch.zeros(1 + config.feature_dim, *([config.grid_resolution] * 3))
    dense[0, xyz[:, 0], xyz[:, 1], xyz[:, 2]] = 1
    dense[1:].permute(1, 2, 3, 0)[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = (feats - mean) / std
    return dense


class ResponseLatentDataset(Dataset):
    def __init__(
        self, samples: list[dict], data: dict, base: Path, model: ModelConfig,
        view_count: int, seed: int, surface_samples: int = 0,
    ) -> None:
        self.samples = samples
        self.model = model
        self.view_count = view_count
        self.seed = seed
        self.surface_samples = surface_samples
        self.epoch = 0
        self.raw_shape = tuple(data["raw_response_shape"])
        if len(self.raw_shape) != 2 or any(type(size) is not int or size <= 0 for size in self.raw_shape):
            raise ValueError("Invalid raw_response_shape.")
        raw_channels = self.raw_shape[0] * self.raw_shape[1]
        indices = data["active_response_indices"]
        if not indices or any(type(index) is not int or not 0 <= index < raw_channels for index in indices):
            raise ValueError("Invalid active_response_indices.")
        if len(set(indices)) != len(indices):
            raise ValueError("active_response_indices must be unique.")
        self.active = torch.tensor(indices, dtype=torch.long)
        self.response_mean = normalization_vector(data["response_mean"], raw_channels, base, "response_mean")
        self.response_std = normalization_vector(data["response_std"], raw_channels, base, "response_std", True)
        self.latent_mean = normalization_vector(data["latent_mean"], model.feature_dim, base, "latent_mean")
        self.latent_std = normalization_vector(data["latent_std"], model.feature_dim, base, "latent_std", True)
        self.response_multiplier = float(data["response_multiplier"])
        self.response_bias = float(data["response_bias"])
        if not np.isfinite([self.response_multiplier, self.response_bias]).all() or self.response_multiplier == 0:
            raise ValueError("Invalid response normalization.")
        self.sensor_scale = float(data["sensor_translation_scale"])
        self.target_scale = float(data["target_translation_scale"])
        if not np.isfinite([self.sensor_scale, self.target_scale]).all() or min(self.sensor_scale, self.target_scale) <= 0:
            raise ValueError("Invalid translation scales.")
        cache_objects = data["cache_objects"]
        if type(cache_objects) is not int or cache_objects < 1:
            raise ValueError("cache_objects must be a positive integer.")
        self.cache_objects = cache_objects
        self._cached_object = None
        self.entries = []
        for object_index, sample in enumerate(samples):
            latents = load_tensor_file(sample["target_latents"])
            if not isinstance(latents, dict) or not isinstance(latents.get("feats"), list) or not latents["feats"]:
                raise ValueError("Invalid target_latents.feats.")
            self.entries.extend((object_index, target_index) for target_index in range(len(latents["feats"])))
        if not self.entries:
            raise ValueError("The dataset split contains no target poses.")

    def _load_object(self, object_index: int) -> tuple:
        sample = self.samples[object_index]
        responses = load_tensor_file(sample["responses"])
        latents = load_tensor_file(sample["target_latents"])
        if not isinstance(responses, dict) or not isinstance(latents, dict):
            raise ValueError("Invalid response or latent file.")
        sensors = require_tensor(responses.get("postures"), "response postures", (None, 6)).float().clone()
        if "sensor_views" in sample:
            views = load_tensor_file(sample["sensor_views"])
            if not isinstance(views, dict) or not torch.equal(views.get("postures", torch.empty(0)).float(), sensors):
                raise ValueError("Sensor-view and response posture orders disagree.")
            if not torch.equal(views.get("sensor_indices", torch.empty(0)), responses.get("sensor_indices", torch.empty(1))):
                raise ValueError("Sensor-view and response indices disagree.")
        raw = responses.get("generated_data")
        if isinstance(raw, Tensor) and raw.ndim == 4 and raw.shape[1] == 1:
            raw = raw[:, 0]
        raw = require_tensor(raw, "generated_data", (len(sensors), *self.raw_shape)).float().flatten(1)
        sensor_indices = responses.get("sensor_indices")
        if sensor_indices is not None:
            sensor_indices = require_tensor(sensor_indices, "sensor_indices", (len(sensors),))
            if not torch.equal(sensor_indices, torch.arange(len(sensors))):
                raise ValueError("Response rows must follow sensor_indices order.")
        normalized = (raw * self.response_multiplier + self.response_bias - self.response_mean) / self.response_std
        normalized = normalized.index_select(1, self.active).contiguous()
        if not torch.isfinite(normalized).all():
            raise ValueError("Response normalization produced non-finite inputs.")
        postures = load_tensor_file(sample["target_postures"])
        if isinstance(postures, dict):
            postures = postures.get("postures")
        targets = require_tensor(postures, "target postures", (None, 6)).float().clone()
        embedded = latents.get("postures")
        if embedded is not None:
            embedded = require_tensor(embedded, "latent postures", tuple(targets.shape))
            if not torch.equal(embedded.float(), targets):
                raise ValueError("Latent and target posture orders disagree.")
        if not isinstance(latents.get("feats"), list) or not isinstance(latents.get("coords"), list):
            raise ValueError("Invalid latent fields.")
        mapping = json.loads(sample["sensor_map"].read_text(encoding="utf-8"))["target_to_sensor_indices"]
        if not isinstance(mapping, list) or not len(targets) == len(mapping) == len(latents["feats"]) == len(latents["coords"]):
            raise ValueError("Target counts do not match.")
        for row in mapping:
            if not isinstance(row, list) or len(row) < self.view_count:
                raise ValueError(f"Every target needs at least {self.view_count} mapped views.")
            if any(type(index) is not int or not 0 <= index < len(sensors) for index in row):
                raise ValueError("Sensor map contains an invalid response row.")
            if len(set(row)) != len(row):
                raise ValueError("Sensor map contains duplicate response rows.")
        sensors[:, 3:] /= self.sensor_scale
        targets[:, 3:] /= self.target_scale
        surface_points = None
        if self.surface_samples:
            surface = load_tensor_file(sample["target_surface_points"])
            if not isinstance(surface, dict):
                raise ValueError("Invalid surface-point file.")
            surface_postures = require_tensor(surface.get("postures"), "surface postures", tuple(postures.shape))
            if not torch.equal(surface_postures.float(), postures.float()):
                raise ValueError("Surface-point and target posture orders disagree.")
            surface_points = require_tensor(surface.get("points"), "surface points", (len(targets), None, 3)).float()
            if surface_points.shape[1] < self.surface_samples:
                raise ValueError("Insufficient target surface points.")
        return normalized, sensors, targets, latents, mapping, surface_points

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cached_object"] = None
        return state

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        if self._cached_object is None:
            self._cached_object = lru_cache(maxsize=self.cache_objects)(self._load_object)
        object_index, target_index = self.entries[index]
        responses, sensors, targets, latents, mapping, surface_points = self._cached_object(object_index)
        candidates = torch.tensor(mapping[target_index], dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
        selected = candidates[torch.randperm(len(candidates), generator=generator)[:self.view_count]]
        poses = relative_postures(sensors[selected].unsqueeze(0), targets[target_index].unsqueeze(0))[0]
        target = sparse_to_dense_target(latents["coords"][target_index], latents["feats"][target_index],
                                        self.model, self.latent_mean, self.latent_std)
        if surface_points is None:
            return responses[selected], poses, target
        point_indices = torch.randperm(surface_points.shape[1], generator=generator)[:self.surface_samples]
        return responses[selected], poses, target, surface_points[target_index, point_indices]


class Proximity3DDataModule(L.LightningDataModule):
    def __init__(
        self, data: dict, base: Path, model: ModelConfig, training: TrainingConfig,
        channel_count: int, loss: LossConfig | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.base = base
        self.model = model
        self.training = training
        self.channel_count = channel_count
        self.surface_samples = loss.surface_samples if loss is not None and loss.uses_decoder else 0
        self.training_dataset = None
        self.validation_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if self.training_dataset is not None:
            return
        if len(self.data["active_response_indices"]) != self.channel_count:
            raise ValueError("Channel count mismatch.")
        surface_root = None
        if self.surface_samples:
            surface_root = resolve_path(self.data.get("target_surface_points", ""), self.base)
        samples = discover_samples(resolve_path(self.data["dataset_config"], self.base), surface_root)
        splits = split_samples(samples, self.data["validation_fraction"], self.data["split_seed"])
        self.training_dataset = ResponseLatentDataset(splits["train"], self.data, self.base, self.model,
                                                     self.training.training_views, self.training.seed, self.surface_samples)
        self.validation_dataset = ResponseLatentDataset(splits["validation"], self.data, self.base, self.model,
                                                       self.training.validation_views, self.training.seed, self.surface_samples)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_dataset, batch_size=self.training.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(self.training.seed),
                          num_workers=self.training.num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_dataset, batch_size=self.training.batch_size, shuffle=False,
                          num_workers=self.training.num_workers, pin_memory=torch.cuda.is_available(),
                          persistent_workers=False)
