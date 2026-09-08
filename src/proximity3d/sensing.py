from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SensingManifold:
    name: str
    positions: Tensor
    frames: Tensor
    neighbors: Tensor
    neighbor_mask: Tensor
    response_channel_indices: Tensor

    @property
    def channel_count(self) -> int:
        return int(self.positions.shape[0])


def rotation_matrix_to_6d(rotation: Tensor) -> Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation must end in [3, 3], received {tuple(rotation.shape)}"
        )
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def _require_sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list.")
    return value


def _load_neighbors(
    rows: list[Any],
    channel_count: int,
) -> tuple[Tensor, Tensor]:
    if len(rows) != channel_count:
        raise ValueError(
            f"neighbors must contain {channel_count} rows, received {len(rows)}."
        )

    parsed: list[list[int]] = []
    for site, row in enumerate(rows):
        values = _require_sequence(row, f"neighbors[{site}]")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError(f"neighbors[{site}] must contain integer site indices.")
        if len(values) != len(set(values)):
            raise ValueError(f"neighbors[{site}] contains duplicate site indices.")
        if site in values:
            raise ValueError(f"neighbors[{site}] must not contain a self-loop.")
        if any(value < 0 or value >= channel_count for value in values):
            raise ValueError(f"neighbors[{site}] contains an out-of-range site index.")
        if not values:
            raise ValueError(f"neighbors[{site}] must not be empty.")
        parsed.append(sorted(values))

    for site, values in enumerate(parsed):
        for neighbor in values:
            if site not in parsed[neighbor]:
                raise ValueError(
                    f"Adjacency must be symmetric: {site} -> {neighbor} has no reverse edge."
                )

    visited = {0}
    frontier = [0]
    while frontier:
        site = frontier.pop()
        for neighbor in parsed[site]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    if len(visited) != channel_count:
        raise ValueError("The sensing-manifold graph must be connected.")

    maximum_degree = max(map(len, parsed))
    neighbor_index = torch.zeros(
        channel_count,
        maximum_degree,
        dtype=torch.long,
    )
    neighbor_mask = torch.zeros(
        channel_count,
        maximum_degree,
        dtype=torch.bool,
    )
    for site, values in enumerate(parsed):
        degree = len(values)
        neighbor_index[site, :degree] = torch.tensor(values, dtype=torch.long)
        neighbor_mask[site, :degree] = True
    return neighbor_index, neighbor_mask


def load_sensing_manifold(artifact: dict[str, Any]) -> SensingManifold:
    if not isinstance(artifact, dict):
        raise TypeError("Invalid manifold configuration.")
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported sensing-manifold schema version.")

    name = artifact.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Missing manifold name.")

    positions = torch.tensor(
        _require_sequence(artifact.get("positions"), "positions"),
        dtype=torch.float32,
    )
    frames = torch.tensor(
        _require_sequence(artifact.get("frames"), "frames"),
        dtype=torch.float32,
    )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape [channels, 3], received {tuple(positions.shape)}."
        )
    channel_count = int(positions.shape[0])
    if channel_count < 2:
        raise ValueError(
            "A sensing manifold requires at least two physical sites, "
            f"received {channel_count}."
        )
    if frames.shape != (channel_count, 3, 3):
        raise ValueError(
            "frames must have shape "
            f"[{channel_count}, 3, 3], received {tuple(frames.shape)}."
        )
    if not torch.isfinite(positions).all() or not torch.isfinite(frames).all():
        raise ValueError("positions and frames must contain only finite values.")
    if torch.unique(positions, dim=0).shape[0] != channel_count:
        raise ValueError("Electrode positions must be unique.")

    identity = torch.eye(3, dtype=frames.dtype).expand(channel_count, -1, -1)
    gram = frames.transpose(-1, -2) @ frames
    if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
        raise ValueError("Non-orthonormal electrode frame.")
    determinants = torch.linalg.det(frames)
    if not torch.allclose(
        determinants,
        torch.ones_like(determinants),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Invalid electrode rotation.")

    response_indices_raw = _require_sequence(
        artifact.get("response_channel_indices"),
        "response_channel_indices",
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in response_indices_raw
    ):
        raise TypeError("response_channel_indices must contain integers.")
    response_indices = torch.tensor(response_indices_raw, dtype=torch.long)
    if response_indices.shape != (channel_count,):
        raise ValueError(
            "response_channel_indices must contain one entry per electrode."
        )
    if sorted(response_indices.tolist()) != list(range(channel_count)):
        raise ValueError(
            "Invalid response_channel_indices."
        )

    neighbors, neighbor_mask = _load_neighbors(
        _require_sequence(artifact.get("neighbors"), "neighbors"),
        channel_count,
    )
    return SensingManifold(
        name=name,
        positions=positions.contiguous(),
        frames=frames.contiguous(),
        neighbors=neighbors,
        neighbor_mask=neighbor_mask,
        response_channel_indices=response_indices,
    )


class _TwoLayerMLP(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )


class ManifoldSensingAttention(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalization = nn.LayerNorm(embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        features: Tensor,
        edge_embedding: Tensor,
        neighbor_index: Tensor,
        neighbor_mask: Tensor,
    ) -> Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"features must have shape [scans, channels, dim], got {tuple(features.shape)}."
            )
        normalized = self.normalization(features)
        query = self.query(normalized)
        keys = self.key(normalized)[:, neighbor_index]
        values = self.value(normalized)[:, neighbor_index]
        messages = edge_embedding.unsqueeze(0)

        logits = (
            query.unsqueeze(2) * (keys + messages)
        ).sum(dim=-1) / math.sqrt(self.embedding_dim)
        logits = logits.masked_fill(~neighbor_mask.unsqueeze(0), -torch.inf)
        attention = torch.softmax(logits, dim=-1)
        update = (
            attention.unsqueeze(-1) * (values + messages)
        ).sum(dim=2)
        return features + update


class _LearnedQueryPooling(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.key = nn.Linear(embedding_dim * 2, embedding_dim)
        self.value = nn.Linear(embedding_dim * 2, embedding_dim)
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, features: Tensor, site_embedding: Tensor) -> Tensor:
        site_embedding = site_embedding.unsqueeze(0).expand(
            features.shape[0],
            -1,
            -1,
        )
        tokens = torch.cat((features, site_embedding), dim=-1)
        keys = self.key(tokens)
        values = self.value(tokens)
        query = self.query.expand(features.shape[0], -1, -1)
        logits = torch.bmm(query, keys.transpose(1, 2))
        logits = logits / math.sqrt(self.embedding_dim)
        attention = torch.softmax(logits, dim=-1)
        return torch.bmm(attention, values).squeeze(1)


class SensingManifoldEncoder(nn.Module):
    def __init__(
        self,
        manifold: SensingManifold,
        *,
        output_dim: int,
        embedding_dim: int,
        num_layers: int,
        scan_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if isinstance(output_dim, bool) or not isinstance(output_dim, int):
            raise TypeError("output_dim must be an integer.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        self.channel_count = manifold.channel_count
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.scan_chunk_size = scan_chunk_size
        self.manifold_name = manifold.name
        self.register_buffer(
            "response_channel_indices",
            manifold.response_channel_indices.clone(),
            persistent=False,
        )
        self.register_buffer(
            "neighbor_index",
            manifold.neighbors.clone(),
            persistent=False,
        )
        self.register_buffer(
            "neighbor_mask",
            manifold.neighbor_mask.clone(),
            persistent=False,
        )

        site_geometry = torch.cat(
            (manifold.positions, rotation_matrix_to_6d(manifold.frames)),
            dim=-1,
        )
        receiver_frames = manifold.frames[:, None].expand(
            -1,
            manifold.neighbors.shape[1],
            -1,
            -1,
        )
        sender_positions = manifold.positions[manifold.neighbors]
        receiver_positions = manifold.positions[:, None]
        relative_position = torch.matmul(
            receiver_frames.transpose(-1, -2),
            (sender_positions - receiver_positions).unsqueeze(-1),
        ).squeeze(-1)
        sender_frames = manifold.frames[manifold.neighbors]
        relative_frame = (
            receiver_frames.transpose(-1, -2) @ sender_frames
        )
        edge_geometry = torch.cat(
            (relative_position, rotation_matrix_to_6d(relative_frame)),
            dim=-1,
        )
        self.register_buffer("site_geometry", site_geometry, persistent=False)
        self.register_buffer("edge_geometry", edge_geometry, persistent=False)

        self.response_embedding = _TwoLayerMLP(1, embedding_dim)
        self.geometry_embedding = _TwoLayerMLP(9, embedding_dim)
        self.edge_embedding = _TwoLayerMLP(9, embedding_dim)
        self.attention_layers = nn.ModuleList(
            ManifoldSensingAttention(embedding_dim)
            for _ in range(num_layers)
        )
        self.site_embedding = _TwoLayerMLP(9, embedding_dim)
        self.pooling = _LearnedQueryPooling(embedding_dim)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def _encode_chunk(
        self,
        responses: Tensor,
        geometry_embedding: Tensor,
        edge_embedding: Tensor,
        site_embedding: Tensor,
    ) -> Tensor:
        features = self.response_embedding(responses.unsqueeze(-1))
        features = features + geometry_embedding.unsqueeze(0)
        for layer in self.attention_layers:
            features = layer(
                features,
                edge_embedding,
                self.neighbor_index,
                self.neighbor_mask,
            )
        pooled = self.pooling(features, site_embedding)
        return self.output_projection(pooled)

    def forward(self, responses: Tensor) -> Tensor:
        if responses.ndim < 2 or responses.shape[-1] != self.channel_count:
            raise ValueError(
                "responses must end in the configured channel dimension "
                f"{self.channel_count}, received {tuple(responses.shape)}."
            )
        output_shape = (*responses.shape[:-1], self.output_dim)
        ordered = responses.index_select(-1, self.response_channel_indices)
        flat = ordered.reshape(-1, self.channel_count)

        geometry_embedding = self.geometry_embedding(self.site_geometry)
        edge_embedding = self.edge_embedding(self.edge_geometry)
        site_embedding = self.site_embedding(self.site_geometry)
        encoded = [
            self._encode_chunk(
                flat[start : start + self.scan_chunk_size],
                geometry_embedding,
                edge_embedding,
                site_embedding,
            )
            for start in range(0, flat.shape[0], self.scan_chunk_size)
        ]
        return torch.cat(encoded, dim=0).reshape(output_shape)
