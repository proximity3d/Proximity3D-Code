from __future__ import annotations

import torch
from torch import Tensor, nn

from proximity3d.config import ModelConfig
from proximity3d.layers import LatentGridDecoder, ShapeTokenEncoder, ViewPoseEncoder
from proximity3d.sensing import SensingManifold, SensingManifoldEncoder


class Proximity3D(nn.Module):
    def __init__(self, manifold: SensingManifold, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.channel_count = manifold.channel_count
        self.sensing_encoder = SensingManifoldEncoder(
            manifold,
            output_dim=config.backbone_dim,
            embedding_dim=config.sensing_dim,
            num_layers=config.sensing_layers,
            scan_chunk_size=config.scan_chunk_size,
        )
        self.pose_encoder = ViewPoseEncoder(
            num_frequencies=config.pose_frequencies, output_dim=config.backbone_dim,
        )
        self.view_projection = nn.Sequential(
            nn.Linear(2 * config.backbone_dim, config.backbone_dim),
            nn.LayerNorm(config.backbone_dim),
            nn.ReLU(inplace=True),
        )
        self.shape_encoder = ShapeTokenEncoder(
            embedding_dim=config.backbone_dim,
            num_heads=config.attention_heads,
            num_layers=config.shape_layers,
            num_register_tokens=config.register_tokens,
            num_shape_tokens=config.shape_tokens,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
            gradient_checkpointing=config.gradient_checkpointing,
            output_normalization=config.shape_output_normalization,
        )
        self.shape_projection = (
            nn.Identity() if config.backbone_dim == config.decoder_dim else nn.Sequential(
                nn.Linear(config.backbone_dim, config.decoder_dim),
                nn.LayerNorm(config.decoder_dim),
            )
        )
        self.latent_decoder = LatentGridDecoder(
            embedding_dim=config.decoder_dim,
            num_heads=config.attention_heads,
            num_layers=config.decoder_layers,
            grid_resolution=config.grid_resolution,
            feature_dim=config.feature_dim,
            gradient_checkpointing=config.gradient_checkpointing,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )

    def encode_views(self, responses: Tensor, relative_poses: Tensor) -> Tensor:
        if responses.ndim != 3 or responses.shape[-1] != self.channel_count:
            raise ValueError(f"responses must have shape [batch, views, {self.channel_count}].")
        if responses.shape[0] == 0 or responses.shape[1] == 0:
            raise ValueError("At least one sample and one view are required.")
        if relative_poses.shape != (*responses.shape[:2], 6):
            raise ValueError("relative_poses must have shape [batch, views, 6].")
        sensing = self.sensing_encoder(responses)
        poses = self.pose_encoder(relative_poses)
        view_tokens = self.view_projection(torch.cat((sensing, poses), dim=-1))
        return self.shape_projection(self.shape_encoder(view_tokens))

    def forward(self, responses: Tensor, relative_poses: Tensor) -> tuple[Tensor, Tensor]:
        return self.latent_decoder(self.encode_views(responses, relative_poses))
