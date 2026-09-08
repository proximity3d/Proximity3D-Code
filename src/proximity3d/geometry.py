from __future__ import annotations

import torch


def _transforms():
    from pytorch3d.transforms import Transform3d
    return Transform3d


def postures_to_transforms(postures: torch.Tensor):
    values = torch.as_tensor(postures)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected postures [N,6], got {tuple(values.shape)}")

    transform_type = _transforms()
    return (
        transform_type(device=values.device, dtype=values.dtype)
        .rotate_axis_angle(values[:, 0], axis="X", degrees=False)
        .rotate_axis_angle(values[:, 1], axis="Y", degrees=False)
        .rotate_axis_angle(values[:, 2], axis="Z", degrees=False)
        .translate(values[:, 3], values[:, 4], values[:, 5])
    )


def rotation_matrices_to_euler_xyz(rotations: torch.Tensor) -> torch.Tensor:
    if rotations.ndim != 3 or tuple(rotations.shape[1:]) != (3, 3):
        raise ValueError(f"Expected rotations [N,3,3], got {tuple(rotations.shape)}")
    sy = rotations[:, 2, 0].clamp(-1.0, 1.0)
    ry = torch.asin(sy)
    rx = torch.atan2(rotations[:, 2, 1], rotations[:, 2, 2])
    rz = torch.atan2(rotations[:, 1, 0], rotations[:, 0, 0])
    singular = sy.abs() >= 0.999999
    rx = torch.where(
        singular,
        torch.atan2(-rotations[:, 1, 2], rotations[:, 1, 1]),
        rx,
    )
    rz = torch.where(singular, torch.zeros_like(rz), rz)
    return torch.stack((rx, ry, rz), dim=1)


def transforms_to_postures(transforms) -> torch.Tensor:
    matrices = transforms.get_matrix()
    rotations = matrices[:, :3, :3].transpose(1, 2)
    translations = matrices[:, 3, :3]
    return torch.cat(
        (rotation_matrices_to_euler_xyz(rotations), translations),
        dim=1,
    )


def relative_postures(
    sensor_postures: torch.Tensor,
    target_postures: torch.Tensor,
) -> torch.Tensor:
    sensors = torch.as_tensor(sensor_postures)
    targets = torch.as_tensor(target_postures, device=sensors.device, dtype=sensors.dtype)
    if sensors.ndim != 3 or sensors.shape[2] != 6:
        raise ValueError(f"Expected sensor postures [B,N,6], got {tuple(sensors.shape)}")
    if targets.ndim != 2 or targets.shape != (sensors.shape[0], 6):
        raise ValueError(
            f"Expected target postures [{sensors.shape[0]},6], got {tuple(targets.shape)}"
        )

    batch_size, view_count, _ = sensors.shape
    expanded_targets = targets[:, None].expand(-1, view_count, -1).reshape(-1, 6)
    sensor_transforms = postures_to_transforms(sensors.reshape(-1, 6))
    target_transforms = postures_to_transforms(expanded_targets)
    relative = target_transforms.inverse().compose(sensor_transforms)
    return transforms_to_postures(relative).reshape(batch_size, view_count, 6)
