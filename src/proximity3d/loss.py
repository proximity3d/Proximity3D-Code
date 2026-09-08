from __future__ import annotations

import torch
from torch import Tensor

from proximity3d.config import LossConfig


def chamfer_l1(first: Tensor, second: Tensor, chunk_size: int = 512) -> Tensor:
    if (
        first.ndim != 2 or second.ndim != 2 or first.shape[1] != 3 or second.shape[1] != 3
        or not len(first) or not len(second)
    ):
        raise ValueError("Invalid Chamfer inputs.")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("Surface points must be finite.")
    first, second = first.float(), second.float()

    def directed(source: Tensor, target: Tensor) -> Tensor:
        with torch.no_grad():
            indices = []
            for block in source.split(chunk_size):
                distances = torch.cdist(block, target, p=2, compute_mode="donot_use_mm_for_euclid_dist")
                indices.append(distances.argmin(dim=1))
            nearest = torch.cat(indices)
        return torch.linalg.vector_norm(source - target[nearest], dim=-1).mean()

    return 0.5 * (directed(first, second) + directed(second, first))


def sample_surface_points(vertices: Tensor, faces: Tensor, count: int) -> Tensor:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not torch.isfinite(vertices).all():
        raise ValueError("Invalid mesh vertices.")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Empty or non-triangular mesh.")
    if faces.dtype not in (torch.int32, torch.int64) or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("Invalid mesh faces.")
    if type(count) is not int or count < 1:
        raise ValueError("Surface sample count must be a positive integer.")
    triangles = vertices.float()[faces.long()]
    with torch.no_grad():
        areas = torch.linalg.vector_norm(
            torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1),
            dim=-1,
        )
        if not torch.isfinite(areas).all() or areas.sum() <= 0:
            raise ValueError("Mesh must have finite positive surface area.")
        selected = torch.multinomial(areas, count, replacement=True)
        random = torch.rand(count, 2, device=vertices.device)
        root = random[:, :1].sqrt()
        weights = torch.cat((1 - root, root * (1 - random[:, 1:]), root * random[:, 1:]), dim=1)
    return (triangles[selected] * weights.unsqueeze(-1)).sum(dim=1)


def reconstruction_loss(
    prediction: tuple[Tensor, Tensor], target: Tensor, config: LossConfig | None = None,
    *, feature_std: Tensor | None = None, decoder_loss: Tensor | None = None,
) -> dict[str, Tensor]:
    config = config or LossConfig()
    logits, features = prediction
    if (
        logits.ndim != 5 or features.ndim != 5 or logits.shape[1] != 1
        or logits.shape[0] != features.shape[0] or logits.shape[2:] != features.shape[2:]
        or target.shape != (logits.shape[0], 1 + features.shape[1], *logits.shape[2:])
    ):
        raise ValueError("Target shape mismatch.")
    probability = logits.float().sigmoid()
    occupancy_target = target[:, :1].float()
    intersection = (probability * occupancy_target).flatten(1).sum(1)
    union = probability.flatten(1).sum(1) + occupancy_target.flatten(1).sum(1) - intersection
    soft_iou = (intersection / (union + 1e-6)).mean()
    occupied = occupancy_target[:, 0] > 0.5
    zero = (logits.float().sum() + features.float().sum()) * 0
    occupancy_loss = 1 - soft_iou if config.occupancy_enabled and config.occupancy_weight > 0 else zero
    feature_loss = zero
    if config.feature_enabled and config.feature_weight > 0:
        difference = features.float() - target[:, 1:].float()
        if feature_std is not None:
            scale = feature_std.to(difference).flatten()
            if scale.numel() not in (1, features.shape[1]) or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("Invalid feature_std.")
            difference = difference * scale.reshape(1, -1, 1, 1, 1)
        squared_distance = difference.square().sum(dim=1)
        per_sample = (squared_distance * occupied).flatten(1).sum(1) / occupied.flatten(1).sum(1).clamp_min(1)
        feature_loss = per_sample.mean()
    if config.uses_decoder:
        if decoder_loss is None or decoder_loss.ndim != 0 or not torch.isfinite(decoder_loss):
            raise ValueError("Invalid decoder loss.")
    else:
        decoder_loss = zero
    predicted_occupied = probability[:, 0] > 0.5
    hard_intersection = (predicted_occupied & occupied).flatten(1).sum(1)
    hard_union = (predicted_occupied | occupied).flatten(1).sum(1)
    return {
        "loss": config.occupancy_weight * occupancy_loss + config.feature_weight * feature_loss
        + config.decoder_weight * decoder_loss,
        "occupancy_loss": occupancy_loss,
        "feature_loss": feature_loss,
        "decoder_loss": decoder_loss,
        "soft_iou": soft_iou,
        "hard_iou": (hard_intersection.float() / hard_union.clamp_min(1)).mean(),
    }
