from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor

from proximity3d.config import LossConfig, ModelConfig, TrainingConfig, load_config, resolve_path
from proximity3d.data import Proximity3DDataModule, normalization_vector
from proximity3d.decoder_loss import FrozenDecoderLoss
from proximity3d.loss import reconstruction_loss
from proximity3d.model import Proximity3D
from proximity3d.sensing import load_sensing_manifold


class TrainingModule(L.LightningModule):
    def __init__(
        self, network: Proximity3D, config: TrainingConfig, sensor: dict, data: dict,
        loss: LossConfig | None = None, base: Path = Path("."),
    ) -> None:
        super().__init__()
        self.network = network
        self.config = config
        self.loss_config = loss or LossConfig()
        self.register_buffer("feature_std", normalization_vector(
            data["latent_std"], network.config.feature_dim, base, "latent_std", True,
        ))
        self.decoder_objective = (
            FrozenDecoderLoss(self.loss_config, network.config, data, base) if self.loss_config.uses_decoder else None
        )
        self.translation_noise = config.translation_noise / data["sensor_translation_scale"]
        self.save_hyperparameters({"model": asdict(network.config), "training": asdict(config),
                                  "sensor": sensor, "data": data, "loss": asdict(self.loss_config)})

    def forward(self, responses: Tensor, relative_poses: Tensor) -> tuple[Tensor, Tensor]:
        return self.network(responses, relative_poses)

    def _noisy_poses(self, poses: Tensor) -> Tensor:
        noisy = poses.clone()
        noisy[..., :3] += (torch.rand_like(poses[..., :3]) * 2 - 1) * math.radians(self.config.rotation_noise_degrees)
        noisy[..., 3:] += (torch.rand_like(poses[..., 3:]) * 2 - 1) * self.translation_noise
        return noisy

    def _log_metrics(self, prefix: str, metrics: dict[str, Tensor], batch_size: int) -> None:
        self.log_dict({f"{prefix}_{key}": value for key, value in metrics.items()},
                      on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def _loss(
        self, prediction: tuple[Tensor, Tensor], target: Tensor, surface_points: Tensor | None,
    ) -> dict[str, Tensor]:
        decoder_loss = None
        if self.decoder_objective is not None:
            decoder_loss = self.decoder_objective(prediction, surface_points)
        return reconstruction_loss(
            prediction, target, self.loss_config, feature_std=self.feature_std, decoder_loss=decoder_loss,
        )

    def training_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        responses, poses, target = batch[:3]
        surface_points = batch[3] if len(batch) == 4 else None
        if responses.shape[1] != self.config.training_views:
            raise ValueError("Training batch has an unexpected view count.")
        noise = torch.randn_like(responses[:, :1]) * self.config.response_noise_std
        responses = responses + noise
        split = int(torch.randint(self.config.minimum_split_views,
                                  self.config.training_views - self.config.minimum_split_views + 1, ()).item())
        first = self._loss(self(responses[:, :split], self._noisy_poses(poses[:, :split])), target, surface_points)
        second = self._loss(self(responses[:, split:], self._noisy_poses(poses[:, split:])), target, surface_points)
        metrics = {name: (first[name] + second[name]) * 0.5 for name in first}
        self._log_metrics("train", metrics, len(responses))
        return metrics["loss"]

    def validation_step(self, batch: tuple[Tensor, ...], batch_idx: int) -> Tensor:
        responses, poses, target = batch[:3]
        surface_points = batch[3] if len(batch) == 4 else None
        if responses.shape[1] != self.config.validation_views:
            raise ValueError("Validation batch has an unexpected view count.")
        metrics = self._loss(self(responses, poses), target, surface_points)
        self._log_metrics("val", metrics, len(responses))
        return metrics["loss"]

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        if self.config.warmup_steps >= total_steps:
            raise ValueError("warmup_steps exceeds training budget.")

        def multiplier(step: int) -> float:
            if step < self.config.warmup_steps:
                return (step + 1) / self.config.warmup_steps
            cosine_steps = max(1, total_steps - self.config.warmup_steps)
            progress = min(1.0, max(0.0, (step - self.config.warmup_steps) / cosine_steps))
            cosine = (1 + math.cos(math.pi * progress)) * 0.5
            return self.config.minimum_lr_ratio + (1 - self.config.minimum_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_epoch_start(self) -> None:
        self.trainer.datamodule.training_dataset.set_epoch(self.current_epoch)

    def on_fit_start(self) -> None:
        seed = self.trainer.strategy.broadcast(self.config.seed, src=0)
        self.config = replace(self.config, seed=seed)
        self.hparams["training"]["seed"] = seed
        data = self.trainer.datamodule
        data.training = replace(data.training, seed=seed)
        data.training_dataset.seed = seed
        data.validation_dataset.seed = seed
        L.seed_everything(seed, workers=True)


def run(config_path: str | Path) -> None:
    content, base = load_config(config_path)
    config = TrainingConfig(**content["training"])
    model_config = ModelConfig(**content["model"])
    loss_config = LossConfig(**content["loss"])
    manifold = load_sensing_manifold(content["sensor"])
    L.seed_everything(config.seed, workers=True)
    data = Proximity3DDataModule(content["data"], base, model_config, config, manifold.channel_count, loss_config)
    data.setup("fit")
    output = resolve_path(config.output_dir, base)
    if "LOGS" not in output.parts:
        raise ValueError("Invalid output_dir: expected LOGS.")
    if (
        int(os.environ.get("LOCAL_RANK", "0")) == 0
        and output.exists()
        and any(output.iterdir())
    ):
        raise FileExistsError("Output directory is not empty.")
    network = Proximity3D(manifold, model_config)
    module = TrainingModule(network, config, content["sensor"], content["data"], loss_config, base)
    trainer = L.Trainer(
        accelerator=config.accelerator, devices=config.devices,
        strategy="ddp" if config.devices > 1 else "auto",
        max_epochs=config.max_epochs, precision=config.precision,
        default_root_dir=output,
        logger=TensorBoardLogger(save_dir=str(output), name="", version=""),
        callbacks=[
            ModelCheckpoint(dirpath=output / "checkpoints", monitor="val_loss", mode="min",
                            save_top_k=3, save_last=True, filename="{epoch:04d}-{val_loss:.6f}"),
            LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
    )
    trainer.fit(module, datamodule=data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Proximity3D from random initialization.")
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
