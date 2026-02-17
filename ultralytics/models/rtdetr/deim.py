# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
from copy import copy
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format
from ultralytics.utils import LOGGER, colorstr

from .detr_augment import compute_policy_epochs, rtdetr_deim_transforms
from .train import RTDETRTrainer
from .val import RTDETRDataset, RTDETRValidator

__all__ = ("RTDETRDEIMDataset", "RTDETRDEIMValidator", "RTDETRDEIMTrainer")


class _RTDETRBatchMixUp:
    """Batch-level MixUp variant used by the DEIM training recipe."""

    def __init__(self, mixup_prob: float, mixup_epochs: tuple[int, int]) -> None:
        self.mixup_prob = mixup_prob
        self.mixup_epochs = mixup_epochs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for MixUp scheduling."""
        self.epoch = epoch

    def __call__(self, batch: list[dict]) -> dict:
        new_batch = YOLODataset.collate_fn(batch)
        start, stop = self.mixup_epochs
        if start <= self.epoch < stop and random.random() < self.mixup_prob:
            new_batch = self._apply_mixup(new_batch)
        return new_batch

    def _apply_mixup(self, batch: dict) -> dict:
        images = batch["img"]
        bs = images.shape[0]
        if bs < 2:
            return batch

        beta = random.uniform(0.45, 0.55)
        shifted_images = torch.roll(images, shifts=1, dims=0)
        batch["img"] = ((1 - beta) * shifted_images + beta * images).to(images.dtype)

        bboxes = batch["bboxes"]
        cls = batch["cls"]
        batch_idx = batch["batch_idx"]
        new_batch_idx = (batch_idx + 1) % bs

        batch["bboxes"] = torch.cat([bboxes, bboxes], dim=0)
        batch["cls"] = torch.cat([cls, cls], dim=0)
        batch["batch_idx"] = torch.cat([batch_idx, new_batch_idx], dim=0)
        return batch


class RTDETRDEIMDataset(RTDETRDataset):
    """RT-DETR dataset variant that uses a dedicated DEIM augmentation pipeline."""

    def __init__(self, *args, data=None, **kwargs):
        hyp = kwargs["hyp"]
        self.policy_epochs, self.mixup_epochs = self._compute_deim_schedule(hyp)
        if not hasattr(hyp, "mosaic"):
            raise AttributeError("RTDETRDEIMDataset requires 'mosaic' hyperparameter in hyp.")
        if not hasattr(hyp, "mixup"):
            raise AttributeError("RTDETRDEIMDataset requires 'mixup' hyperparameter in hyp.")
        self.mosaic_prob = float(hyp.mosaic)
        self.mixup_prob = float(hyp.mixup)
        super().__init__(*args, data=data, **kwargs)
        if self.augment:
            if self.mixup_prob > 0.0:
                self.collate_fn = _RTDETRBatchMixUp(mixup_prob=self.mixup_prob, mixup_epochs=self.mixup_epochs)
            self.set_epoch(0)

    def _compute_deim_schedule(self, hyp) -> tuple[tuple[int, int, int], tuple[int, int]]:
        """Compute DEIM stage boundaries from epochs only."""
        policy_epochs = compute_policy_epochs(hyp)
        return policy_epochs, policy_epochs[:2]

    def build_transforms(self, hyp=None):
        """Build DEIM transforms for train and standard formatting for train/val."""
        if self.augment:
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = rtdetr_deim_transforms(
                self,
                self.imgsz,
                hyp,
                stretch=True,
                policy_epochs=self.policy_epochs,
                mosaic_prob=self.mosaic_prob,
            )
        else:
            transforms = Compose([])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to transforms and collate_fn for DEIM stage scheduling."""
        self.epoch = epoch
        if hasattr(self.transforms, "set_epoch"):
            self.transforms.set_epoch(epoch)
        collate_fn = getattr(self, "collate_fn", None)
        if collate_fn is not None and hasattr(collate_fn, "set_epoch"):
            collate_fn.set_epoch(epoch)


class RTDETRDEIMValidator(RTDETRValidator):
    """Validator that builds the DEIM dataset variant."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )


class RTDETRDEIMTrainer(RTDETRTrainer):
    """RT-DETR trainer variant with isolated DEIM augmentation scheduling."""

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        dataset = RTDETRDEIMDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )
        return dataset

    def _on_train_epoch_start(self, trainer=None):
        """Apply DEIM epoch scheduling to transforms/collate and stop multi-scale at stage-4 start."""
        trainer = trainer or self
        epoch = trainer.epoch
        dataset = trainer.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        if not hasattr(dataset, "policy_epochs"):
            raise AttributeError("RTDETRDEIMTrainer requires dataset.policy_epochs for DEIM scheduling.")
        stop_epoch = int(dataset.policy_epochs[-1])
        if epoch == stop_epoch and trainer.args.multi_scale > 0:
            trainer.args.multi_scale = 0.0
            LOGGER.info(f"DEIM stage-4 at epoch {epoch}: disabling multi-scale")

    def train(self, *args, **kwargs):
        # DEIM trainer handles augmentation schedule explicitly.
        # Keep base trainer's close_mosaic hook disabled to avoid overriding DEIM policy.
        if self.args.close_mosaic:
            self.args.close_mosaic = 0
        if not getattr(self, "_deim_callback_registered", False):
            self.add_callback("on_train_epoch_start", self._on_train_epoch_start)
            self._deim_callback_registered = True
        return super().train(*args, **kwargs)

    def get_validator(self):
        loss_names = ["giou_loss", "cls_loss", "l1_loss"]
        loss_gain = self.model_yaml.get("loss", {}).get("loss_gain", {})
        if "fgl" in loss_gain:
            loss_names.append("fgl_loss")
        if "ddf" in loss_gain:
            loss_names.append("ddf_loss")
        model = getattr(self.model, "module", self.model)
        if getattr(model.model[-1], "one_to_many_groups", 0) > 0:
            loss_names.extend(["giou_o2m", "cls_o2m", "l1_o2m"])
        self.loss_names = tuple(loss_names)
        return RTDETRDEIMValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
