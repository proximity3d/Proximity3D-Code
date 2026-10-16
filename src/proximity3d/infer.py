from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from proximity3d.config import ModelConfig, load_config, resolve_path
from proximity3d.data import normalization_vector, require_tensor
from proximity3d.geometry import relative_postures
from proximity3d.model import Proximity3D
from proximity3d.sensing import load_sensing_manifold


def resolve_object(config_path: Path, object_id: str) -> dict[str, Path]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parts = object_id.replace("\\", "/").split("/")
    if len(parts) != 2 or any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise ValueError("Invalid object ID.")
    label, name = parts
    if label not in config["label"]:
        raise ValueError("Unknown dataset label.")
    return {
        "responses": resolve_path(config["responses"], config_path.parent) / label / f"{name}.pt",
        "target_postures": resolve_path(config["target_postures"], config_path.parent) / label / f"{name}_posture.pt",
        "sensor_map": resolve_path(config["sensor_maps"], config_path.parent) / label / f"{name}.json",
    }


def output_path(args: argparse.Namespace, target_index: int) -> Path:
    label, name = args.object_id.replace("\\", "/").split("/")
    return args.output_dir / label / f"{name}_target{target_index:04d}.ply"


def load_network(path: Path, content: dict) -> Proximity3D:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("network.") for key in state):
        state = {key.removeprefix("network."): value for key, value in state.items() if key.startswith("network.")}
    network = Proximity3D(load_sensing_manifold(content["sensor"]), ModelConfig(**content["model"]))
    network.load_state_dict(state, strict=True)
    return network.eval().requires_grad_(False)


def load_inputs(args: argparse.Namespace, data: dict, base: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    config_path = resolve_path(data["dataset_config"], base)
    sample = resolve_object(config_path, args.object_id)
    response = torch.load(sample["responses"], map_location="cpu", weights_only=True)
    sensors = require_tensor(response["postures"], "postures", (None, 6)).float().clone()
    raw = response["generated_data"]
    if raw.ndim == 4 and raw.shape[1] == 1:
        raw = raw[:, 0]
    raw = require_tensor(raw, "generated_data", (len(sensors), *data["raw_response_shape"])).float().flatten(1)
    if "sensor_indices" in response and not torch.equal(response["sensor_indices"], torch.arange(len(sensors))):
        raise ValueError("Invalid sensor order.")
    targets = torch.load(sample["target_postures"], map_location="cpu", weights_only=True)
    if isinstance(targets, dict):
        targets = targets["postures"]
    targets = require_tensor(targets, "target postures", (None, 6)).float().clone()
    generator = torch.Generator().manual_seed(time.time_ns() % (2**63))
    target_index = torch.randint(len(targets), (), generator=generator).item()
    mapping = json.loads(sample["sensor_map"].read_text(encoding="utf-8"))["target_to_sensor_indices"]
    row = mapping[target_index]
    if len(mapping) != len(targets) or not row or any(type(i) is not int or not 0 <= i < len(sensors) for i in row):
        raise ValueError("Invalid sensor map.")
    if len(set(row)) != len(row) or len(row) < 64:
        raise ValueError("Invalid view selection.")
    selected = torch.tensor(row)[torch.randperm(len(row), generator=generator)[:64]]
    mean = normalization_vector(data["response_mean"], raw.shape[1], base, "response_mean")
    std = normalization_vector(data["response_std"], raw.shape[1], base, "response_std", True)
    normalized = (raw[selected] * data["response_multiplier"] + data["response_bias"] - mean) / std
    normalized = normalized.index_select(1, torch.tensor(data["active_response_indices"], dtype=torch.long))
    if not torch.isfinite(normalized).all():
        raise ValueError("Invalid normalized responses.")
    sensors[:, 3:] /= data["sensor_translation_scale"]
    targets[:, 3:] /= data["target_translation_scale"]
    poses = relative_postures(sensors[selected].unsqueeze(0), targets[target_index].unsqueeze(0))
    return normalized.unsqueeze(0), poses, target_index


def load_decoder(prefix: Path, resolution: int, device: torch.device):
    from safetensors.torch import load_file
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeDecoder

    descriptor = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    if descriptor["name"] != "FlexiDualGridVaeDecoder":
        raise ValueError("Invalid decoder.")
    decoder = FlexiDualGridVaeDecoder(**descriptor["args"])
    decoder.load_state_dict(load_file(str(prefix) + ".safetensors"), strict=True)
    decoder.set_resolution(resolution)
    return decoder.to(device).eval().requires_grad_(False)


def decode_prediction(prediction: tuple[torch.Tensor, torch.Tensor], decoder, data: dict, base: Path, threshold: float):
    from trellis2.modules.sparse import SparseTensor

    logits, features = prediction
    occupied = logits[0, 0].float().sigmoid() > threshold
    if not occupied.any():
        raise ValueError("Empty predicted occupancy.")
    xyz = occupied.nonzero().to(torch.int32)
    coords = torch.cat((torch.zeros_like(xyz[:, :1]), xyz), dim=1)
    mean = normalization_vector(data["latent_mean"], features.shape[1], base, "latent_mean").to(features.device)
    std = normalization_vector(data["latent_std"], features.shape[1], base, "latent_std", True).to(features.device)
    values = features[0].float().permute(1, 2, 3, 0)[occupied] * std + mean
    mesh = decoder(SparseTensor(feats=values.contiguous(), coords=coords.contiguous()))[0]
    return mesh.vertices, mesh.faces


def export_mesh(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    vertices = require_tensor(vertices, "vertices", (None, 3)).detach().cpu()
    faces = require_tensor(faces, "faces", (None, 3)).detach().cpu()
    if not len(vertices) or not len(faces):
        raise ValueError("Empty mesh.")
    if faces.dtype not in (torch.int32, torch.int64) or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError("Invalid faces.")
    if path.suffix.lower() not in {".ply", ".obj"}:
        raise ValueError("Unsupported output format.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        if path.suffix.lower() == ".ply":
            stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(vertices)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\n")
            stream.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
            for x, y, z in vertices.tolist():
                stream.write(f"{x:.9g} {y:.9g} {z:.9g}\n")
            for a, b, c in faces.tolist():
                stream.write(f"3 {a} {b} {c}\n")
        else:
            for x, y, z in vertices.tolist():
                stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
            for a, b, c in faces.tolist():
                stream.write(f"f {a + 1} {b + 1} {c + 1}\n")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "TRELLIS.2"))
    content, base = load_config(args.config)
    model_config = ModelConfig(**content["model"])
    if model_config.feature_dim != 32 or model_config.grid_resolution != 8:
        raise ValueError("Incompatible decoder dimensions.")
    responses, poses, target_index = load_inputs(args, content["data"], base)
    output = output_path(args, target_index)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device("cuda")
    network = load_network(args.checkpoint, content).to(device)
    prediction = network(responses.to(device), poses.to(device))
    del network
    torch.cuda.empty_cache()
    decoder = load_decoder(Path(__file__).resolve().parents[2] / "checkpoints/shape_dec_next_dc_f16c32_fp16", 128, device)
    vertices, faces = decode_prediction(prediction, decoder, content["data"], base, 0.5)
    export_mesh(output, vertices, faces)
    print(f"Saved {output} ({len(vertices)} vertices, {len(faces)} faces)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a mesh from capacitive responses.")
    parser.add_argument("--config", type=Path, default=Path("configs/proximity3d_384.toml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--object-id", default="models/9477c1ca441f064eef2edac1d39d57fa4550a32ee6c020d37933a0a24b04a567")
    parser.add_argument("--output-dir", type=Path, default=Path("LOGS/inference"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
