# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
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


class _RTDETRDEIMBatchAugment:
    """Batch-level DEIM augmentations (MixUp + CopyBlend) applied in collate_fn."""

    _COPYBLEND_AREA_THRESHOLD = 100.0
    _COPYBLEND_NUM_OBJECTS = 3
    _COPYBLEND_WITH_EXPAND = True
    _COPYBLEND_EXPAND_RATIOS = (0.1, 0.25)

    def __init__(
        self,
        mixup_prob: float,
        mixup_epochs: tuple[int, int],
        copyblend_prob: float,
        copyblend_epochs: tuple[int, int],
    ) -> None:
        self.mixup_prob = mixup_prob
        self.mixup_epochs = mixup_epochs
        self.copyblend_prob = copyblend_prob
        self.copyblend_epochs = copyblend_epochs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for DEIM batch augmentation scheduling."""
        self.epoch = epoch

    def __call__(self, batch: list[dict]) -> dict:
        new_batch = YOLODataset.collate_fn(batch)
        mixup_start, mixup_stop = self.mixup_epochs
        if mixup_start <= self.epoch < mixup_stop and random.random() < self.mixup_prob:
            new_batch = self._apply_mixup(new_batch)
        copyblend_start, copyblend_stop = self.copyblend_epochs
        if copyblend_start <= self.epoch < copyblend_stop and random.random() < self.copyblend_prob:
            new_batch = self._apply_copyblend(new_batch)
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

    @staticmethod
    def _xywhn_to_xyxy(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Convert normalized xywh boxes to absolute xyxy."""
        scale = boxes.new_tensor([w, h, w, h])
        boxes_abs = boxes * scale
        half_wh = boxes_abs[:, 2:4] / 2
        out = boxes_abs.clone()
        out[:, 0:2] = boxes_abs[:, 0:2] - half_wh
        out[:, 2:4] = boxes_abs[:, 0:2] + half_wh
        return out

    @staticmethod
    def _xyxy_to_xywhn(box: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Convert one absolute xyxy box to normalized xywh."""
        bw = (box[2] - box[0]).clamp(min=1.0)
        bh = (box[3] - box[1]).clamp(min=1.0)
        cx = box[0] + bw / 2
        cy = box[1] + bh / 2
        return torch.stack((cx / w, cy / h, bw / w, bh / h)).clamp(0.0, 1.0).to(dtype=torch.float32)

    def _apply_copyblend(self, batch: dict) -> dict:
        """Copy small objects across images in the same batch with blended paste."""
        images = batch["img"]
        bs = images.shape[0]
        if bs < 2:
            return batch

        _, h, w = images.shape[1:]
        bboxes = batch["bboxes"]
        cls = batch["cls"]
        batch_idx = batch["batch_idx"]

        new_boxes, new_cls, new_batch_idx = [], [], []
        device = bboxes.device

        for dst_i in range(bs):
            src_candidates = [i for i in range(bs) if i != dst_i]
            src_i = random.choice(src_candidates)

            src_mask = batch_idx == src_i
            if not src_mask.any():
                continue

            src_boxes_xyxy = self._xywhn_to_xyxy(bboxes[src_mask].to(dtype=torch.float32), w=w, h=h)
            src_cls = cls[src_mask]
            areas = (src_boxes_xyxy[:, 2] - src_boxes_xyxy[:, 0]).clamp(min=0) * (
                src_boxes_xyxy[:, 3] - src_boxes_xyxy[:, 1]
            ).clamp(min=0)
            object_indices = torch.nonzero(areas <= self._COPYBLEND_AREA_THRESHOLD).squeeze(1)
            if object_indices.numel() == 0:
                continue

            num_objects = min(self._COPYBLEND_NUM_OBJECTS, int(object_indices.numel()))
            selected = random.sample(object_indices.tolist(), k=num_objects)

            for obj_idx in selected:
                x1, y1, x2, y2 = src_boxes_xyxy[obj_idx].tolist()
                bw = max(x2 - x1, 1.0)
                bh = max(y2 - y1, 1.0)

                if self._COPYBLEND_WITH_EXPAND:
                    expand_ratio = random.uniform(*self._COPYBLEND_EXPAND_RATIOS)
                    padw = bw * expand_ratio
                    padh = bh * expand_ratio
                else:
                    padw, padh = 0.0, 0.0

                sx1 = max(0, min(w - 1, math.floor(x1 - padw)))
                sy1 = max(0, min(h - 1, math.floor(y1 - padh)))
                sx2 = max(sx1 + 1, min(w, math.ceil(x2 + padw)))
                sy2 = max(sy1 + 1, min(h, math.ceil(y2 + padh)))
                patch_h, patch_w = sy2 - sy1, sx2 - sx1
                if patch_h <= 0 or patch_w <= 0:
                    continue

                max_x = w - patch_w
                max_y = h - patch_h
                if max_x < 0 or max_y < 0:
                    continue
                dx1 = random.randint(0, max_x)
                dy1 = random.randint(0, max_y)
                dx2, dy2 = dx1 + patch_w, dy1 + patch_h

                src_patch = images[src_i, :, sy1:sy2, sx1:sx2]
                dst_patch = images[dst_i, :, dy1:dy2, dx1:dx2]
                alpha = random.uniform(0.45, 0.55)
                blended = (1.0 - alpha) * dst_patch.to(torch.float32) + alpha * src_patch.to(torch.float32)
                if images.dtype.is_floating_point:
                    images[dst_i, :, dy1:dy2, dx1:dx2] = blended.to(images.dtype)
                else:
                    images[dst_i, :, dy1:dy2, dx1:dx2] = blended.round().to(images.dtype)

                ox1 = float(dx1 + (x1 - sx1))
                oy1 = float(dy1 + (y1 - sy1))
                ox2 = ox1 + bw
                oy2 = oy1 + bh
                ox1 = max(0.0, min(ox1, w - 2.0))
                oy1 = max(0.0, min(oy1, h - 2.0))
                ox2 = max(ox1 + 1.0, min(ox2, w - 1.0))
                oy2 = max(oy1 + 1.0, min(oy2, h - 1.0))
                new_box = torch.tensor(
                    [ox1, oy1, ox2, oy2],
                    device=device,
                    dtype=torch.float32,
                )
                new_boxes.append(self._xyxy_to_xywhn(new_box, w=w, h=h).to(device=device, dtype=bboxes.dtype))
                new_cls.append(src_cls[obj_idx].to(device=device, dtype=cls.dtype))
                new_batch_idx.append(torch.tensor(float(dst_i), device=device, dtype=batch_idx.dtype))

        if new_boxes:
            batch["img"] = images
            batch["bboxes"] = torch.cat([bboxes, torch.stack(new_boxes, dim=0)], dim=0)
            batch["cls"] = torch.cat([cls, torch.stack(new_cls, dim=0)], dim=0)
            batch["batch_idx"] = torch.cat([batch_idx, torch.stack(new_batch_idx, dim=0)], dim=0)
        return batch


class RTDETRDEIMDataset(RTDETRDataset):
    """RT-DETR dataset variant that uses a dedicated DEIM augmentation pipeline."""

    def __init__(self, *args, data=None, **kwargs):
        hyp = kwargs["hyp"]
        self.policy_epochs, self.mixup_epochs, self.copyblend_epochs = self._compute_deim_schedule(hyp)
        if not hasattr(hyp, "mosaic"):
            raise AttributeError("RTDETRDEIMDataset requires 'mosaic' hyperparameter in hyp.")
        if not hasattr(hyp, "mixup"):
            raise AttributeError("RTDETRDEIMDataset requires 'mixup' hyperparameter in hyp.")
        if not hasattr(hyp, "copy_paste"):
            raise AttributeError("RTDETRDEIMDataset requires 'copy_paste' hyperparameter in hyp.")
        self.mosaic_prob = float(hyp.mosaic)
        self.mixup_prob = float(hyp.mixup)
        self.copyblend_prob = float(hyp.copy_paste)
        super().__init__(*args, data=data, **kwargs)
        if self.augment:
            if self.mixup_prob > 0.0 or self.copyblend_prob > 0.0:
                self.collate_fn = _RTDETRDEIMBatchAugment(
                    mixup_prob=self.mixup_prob,
                    mixup_epochs=self.mixup_epochs,
                    copyblend_prob=self.copyblend_prob,
                    copyblend_epochs=self.copyblend_epochs,
                )
            self.set_epoch(0)

    def _compute_deim_schedule(self, hyp) -> tuple[tuple[int, int, int], tuple[int, int], tuple[int, int]]:
        """Compute DEIM stage boundaries from epochs only."""
        policy_epochs = compute_policy_epochs(hyp)
        mixup_epochs = policy_epochs[:2]
        copyblend_epochs = (policy_epochs[0], policy_epochs[2])
        return policy_epochs, mixup_epochs, copyblend_epochs

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
