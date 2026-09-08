from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


class ViewPoseEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        num_frequencies: int = 10,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = output_dim
        self.register_buffer(
            "frequency_bands",
            2 ** torch.arange(num_frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim * 2 * num_frequencies, output_dim),
            nn.ReLU(),
        )

    def forward(self, pose: Tensor) -> Tensor:
        if pose.shape[-1] != self.input_dim:
            raise ValueError(
                f"pose must end in dimension {self.input_dim}, got {tuple(pose.shape)}."
            )
        projected = pose.unsqueeze(-1) * self.frequency_bands.to(dtype=pose.dtype)
        fourier = torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)
        flat = fourier.reshape(-1, self.input_dim * 2 * self.num_frequencies)
        return self.projection(flat).reshape(*pose.shape[:-1], self.output_dim)


class ShapeTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        num_register_tokens: int,
        num_shape_tokens: int,
        feedforward_dim: int,
        dropout: float,
        gradient_checkpointing: bool,
        output_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.num_shape_tokens = num_shape_tokens
        self.shape_tokens = nn.Parameter(
            torch.zeros(1, num_shape_tokens, embedding_dim)
        )
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.normalization = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_normalization = (
            nn.LayerNorm(embedding_dim) if output_normalization else nn.Identity()
        )

    def forward(self, view_tokens: Tensor) -> Tensor:
        batch_size = view_tokens.shape[0]
        shape_tokens = self.shape_tokens.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        encoded = torch.cat((shape_tokens, register_tokens, view_tokens), dim=1)
        for layer in self.transformer.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                encoded = activation_checkpoint(layer, encoded, use_reentrant=False)
            else:
                encoded = layer(encoded)
        encoded = self.normalization(encoded)
        projected = self.output_projection(encoded[:, : self.num_shape_tokens])
        return self.output_normalization(projected)


class LatentGridDecoder(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        grid_resolution: int = 8,
        feature_dim: int = 32,
        gradient_checkpointing: bool = False,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.grid_resolution = grid_resolution
        self.feature_dim = feature_dim
        self.gradient_checkpointing = gradient_checkpointing
        query_count = grid_resolution**3
        self.cell_queries = nn.Parameter(
            torch.randn(1, query_count, embedding_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        self.cross_attention = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )


        self.cell_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1 + feature_dim),
        )

    def _cross_attend(self, queries: Tensor, shape_tokens: Tensor) -> Tensor:
        if not (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return self.cross_attention(tgt=queries, memory=shape_tokens)

        decoded = queries
        for layer in self.cross_attention.layers:
            decoded = activation_checkpoint(
                layer,
                decoded,
                shape_tokens,
                use_reentrant=False,
            )
        if self.cross_attention.norm is not None:
            decoded = self.cross_attention.norm(decoded)
        return decoded

    def forward(self, shape_tokens: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = shape_tokens.shape[0]
        queries = self.cell_queries.expand(batch_size, -1, -1)
        decoded = self._cross_attend(queries, shape_tokens)
        cells = self.cell_projection(decoded).reshape(
            batch_size,
            self.grid_resolution,
            self.grid_resolution,
            self.grid_resolution,
            1 + self.feature_dim,
        )
        occupancy = cells[..., :1].permute(0, 4, 1, 2, 3).contiguous()
        features = cells[..., 1:].permute(0, 4, 1, 2, 3).contiguous()
        return occupancy, features
