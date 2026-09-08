from __future__ import annotations

import json
import random
from pathlib import Path

from proximity3d.config import resolve_path


def discover_samples(config_path: Path, surface_root: Path | None = None) -> list[dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") == "proximity3d.consolidated_dataset":
        samples = discover_consolidated(config, config_path.parent)
    elif "format" in config:
        raise ValueError(f"Unsupported dataset format: {config['format']}")
    else:
        samples = discover_legacy(config, config_path.parent)
    names, paths = set(), set()
    for sample in samples:
        if sample["name"] in names or sample["responses"] in paths:
            raise ValueError(f"Duplicate dataset object: {sample['name']}")
        names.add(sample["name"])
        paths.add(sample["responses"])
        if surface_root is not None:
            sample["target_surface_points"] = surface_root / (sample["name"] + ".pt")
        for key in ("responses", "target_postures", "target_latents", "sensor_map", "sensor_views", "target_surface_points"):
            if key in sample and not sample[key].is_file():
                raise FileNotFoundError(sample[key])
    if not samples:
        raise ValueError("Dataset discovery found no objects.")
    return samples


def require_label(value: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or any(
        part in {".", "..", ""} for part in value.replace("\\", "/").split("/")
    ):
        raise ValueError("Invalid dataset label.")
    return value


def discover_legacy(config: dict, base: Path) -> list[dict]:
    keys = {"responses", "target_postures", "target_latents", "sensor_maps", "label"}
    if set(config) != keys:
        raise ValueError("Invalid dataset config fields.")
    roots = {key: resolve_path(config[key], base) for key in keys - {"label"}}
    labels = config["label"]
    if not isinstance(labels, list) or not labels:
        raise ValueError("label must be a non-empty list.")
    samples = []
    for value in labels:
        label = require_label(value)
        directory = roots["responses"] / label
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(directory.glob("*.pt"))
        if not files:
            raise ValueError(f"No response files found in {directory}.")
        for response in files:
            name = response.stem
            samples.append({
                "name": f"{label}/{name}", "dataset": label, "split": None,
                "responses": response.resolve(),
                "target_postures": roots["target_postures"] / label / f"{name}_posture.pt",
                "target_latents": roots["target_latents"] / label / f"{name}_latent.pt",
                "sensor_map": roots["sensor_maps"] / label / f"{name}.json",
            })
    return samples


def discover_consolidated(config: dict, base: Path) -> list[dict]:
    if set(config) != {"format", "schema_version", "roots"} or config["schema_version"] != 1:
        raise ValueError("Unsupported consolidated dataset schema.")
    if not isinstance(config["roots"], list) or not config["roots"]:
        raise ValueError("roots must be a non-empty list.")
    samples = []
    for entry in config["roots"]:
        if not isinstance(entry, dict) or not {"path", "label"} <= entry.keys() or entry.keys() - {"path", "label", "split"}:
            raise ValueError("Invalid dataset root.")
        label = require_label(entry["label"])
        split = entry.get("split")
        if split not in {None, "train", "validation"}:
            raise ValueError("split must be train or validation.")
        root = resolve_path(entry["path"], base)
        if not root.is_dir():
            raise FileNotFoundError(root)
        if (root / "objects").is_dir():
            root = root / "objects"
        manifests = sorted(root.rglob("pipeline_manifest.json"))
        if not manifests:
            raise ValueError(f"No consolidated objects found in {root}.")
        for manifest in manifests:
            directory = manifest.parent.resolve()
            relative = directory.relative_to(root).as_posix()
            name = f"{label}/{directory.name if relative == '.' else relative}"
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            mesh_id = metadata.get("inputs", {}).get("mesh_sha256")
            if not isinstance(mesh_id, str) or not mesh_id:
                raise ValueError(f"Missing source mesh identity in {manifest}.")
            samples.append({
                "name": name, "dataset": label, "split": split, "mesh_id": mesh_id,
                "responses": directory / "responses.pt",
                "target_postures": directory / "target_poses.pt",
                "target_latents": directory / "target_latents.pt",
                "sensor_map": directory / "sensor_map.json",
                "sensor_views": directory / "sensor_views.pt",
            })
    return samples


def split_samples(samples: list[dict], validation_fraction: float, seed: int) -> dict[str, list[dict]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation_fraction.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Invalid split_seed.")
    mesh_groups = {}
    for sample in sorted(samples, key=lambda item: item["name"]):
        mesh_groups.setdefault(sample.get("mesh_id", sample["name"]), []).append(sample)
    if len(mesh_groups) < 2:
        raise ValueError("Insufficient distinct objects.")
    declared = {sample["split"] for sample in samples}
    if declared != {None}:
        if declared != {"train", "validation"}:
            raise ValueError("Incomplete split assignments.")
        if any(len({sample["split"] for sample in group}) != 1 for group in mesh_groups.values()):
            raise ValueError("Source mesh overlaps splits.")
        return {name: [sample for sample in samples if sample["split"] == name] for name in declared}
    grouped = {}
    for group in mesh_groups.values():
        labels = tuple(sorted({sample["dataset"] for sample in group}))
        grouped.setdefault(labels, []).append(group)
    generator = random.Random(seed)
    training, validation = [], []
    for labels in sorted(grouped):
        groups = sorted(grouped[labels], key=lambda group: tuple(sample["name"] for sample in group))
        generator.shuffle(groups)
        count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
        validation.extend(groups[:count])
        training.extend(groups[count:])
    if not validation:
        validation.append(training.pop())
    return {"train": [sample for group in training for sample in group],
            "validation": [sample for group in validation for sample in group]}
