from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from proximity3d.config import LossConfig, ModelConfig, resolve_path
from proximity3d.data import normalization_vector
from proximity3d.loss import chamfer_l1, sample_surface_points


class FrozenDecoderLoss(nn.Module):
    def __init__(self, config: LossConfig, model: ModelConfig, data: dict, base: Path) -> None:
        super().__init__()
        self.config = config
        if model.feature_dim != 32 or model.grid_resolution != 8:
            raise ValueError("Incompatible decoder dimensions.")
        config_path = resolve_path(config.decoder_config, base)
        weights_path = resolve_path(config.decoder_weights, base)
        from safetensors.torch import load_file
        from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder
        descriptor = json.loads(config_path.read_text(encoding="utf-8"))
        if descriptor.get("name") != "FlexiDualGridVaeDecoder" or not isinstance(descriptor.get("args"), dict):
            raise ValueError("Invalid decoder_config.")
        self.decoder = FlexiDualGridVaeDecoder(**descriptor["args"])
        self.decoder.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        self.decoder.set_resolution(config.decoder_resolution)
        self.decoder.requires_grad_(False).eval()
        self.register_buffer("latent_mean", normalization_vector(data["latent_mean"], 32, base, "latent_mean"))
        self.register_buffer("latent_std", normalization_vector(data["latent_std"], 32, base, "latent_std", True))

    def train(self, mode: bool = True) -> FrozenDecoderLoss:
        super().train(mode)
        self.decoder.eval()
        return self

    def _decode_mesh(self, features: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        from o_voxel.convert import flexible_dual_grid_to_mesh
        from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
        from trellis2.modules.sparse import SparseTensor

        latent = SparseTensor(feats=features.contiguous(), coords=coordinates.contiguous())
        decoded = SparseUnetVaeDecoder.forward(self.decoder, latent)
        vertices = (1 + 2 * self.decoder.voxel_margin) * decoded.feats[:, :3].sigmoid() - self.decoder.voxel_margin
        return flexible_dual_grid_to_mesh(
            decoded.coords[:, 1:], vertices, decoded.feats[:, 3:6] > 0,
            F.softplus(decoded.feats[:, 6:7]),
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            grid_size=self.config.decoder_resolution, train=True,
        )

    def forward(self, prediction: tuple[Tensor, Tensor], target_points: Tensor) -> Tensor:
        logits, features = prediction
        if (
            target_points.ndim != 3 or target_points.shape[0] != logits.shape[0]
            or target_points.shape[2] != 3 or not target_points.shape[1]
        ):
            raise ValueError("Invalid target surface points.")
        losses = []
        with torch.autocast(device_type=features.device.type, enabled=False):
            for index in range(len(features)):
                active = logits[index, 0].float().sigmoid() > 0.5
                if not active.any():
                    raise ValueError("Empty predicted occupancy.")
                xyz = active.nonzero().to(torch.int32)
                coordinates = torch.cat((torch.zeros_like(xyz[:, :1]), xyz), dim=1)
                raw = features[index].float().permute(1, 2, 3, 0)[active] * self.latent_std + self.latent_mean
                vertices, faces = self._decode_mesh(raw, coordinates)
                if torch.is_grad_enabled() and features.requires_grad and not vertices.requires_grad:
                    raise RuntimeError("Detached decoder output.")
                predicted_points = sample_surface_points(vertices, faces, self.config.surface_samples)
                losses.append(chamfer_l1(predicted_points, target_points[index], self.config.chamfer_chunk_size))
        return torch.stack(losses).mean()
