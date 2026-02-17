# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import Compose
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.instance import Instances


class _RTDETRToTvTensors:
    def __init__(self) -> None:
        from torchvision import tv_tensors

        self._tv_tensors = tv_tensors

    @staticmethod
    def _build_labels_tensor(cls: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(cls.reshape(-1), dtype=torch.int64) if len(cls) else torch.zeros((0,), dtype=torch.int64)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels.pop("img")
        instances = labels.pop("instances", None)
        cls = labels.pop("cls")

        h, w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 3:
            image = Image.fromarray(img[..., ::-1].copy())  # BGR -> RGB
        else:
            image = Image.fromarray(img.copy())

        if instances is None or len(instances) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            instances.convert_bbox(format="xyxy")
            instances.denormalize(w, h)
            boxes_tensor = torch.as_tensor(instances.bboxes, dtype=torch.float32)

        labels["image"] = image
        labels["boxes"] = self._tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(h, w))
        labels["labels"] = self._build_labels_tensor(cls)
        return labels


class _RTDETRFromTvTensors:
    @staticmethod
    def _to_numpy_image(image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            if img.ndim == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()
        else:
            img = np.asarray(image)
        if np.issubdtype(img.dtype, np.floating):
            if img.size and img.max() <= 1.0:
                img = img * 255.0
            img = img.round().astype(np.uint8)
        return img

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        image = labels.pop("image")
        boxes_t = labels.pop("boxes", None)
        labels_t = labels.pop("labels", None)

        img_np = self._to_numpy_image(image)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = img_np[..., ::-1]  # RGB -> BGR

        if boxes_t is None or boxes_t.numel() == 0:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls_out = np.zeros((0, 1), dtype=np.int64)
        else:
            bboxes = boxes_t.to(torch.float32).cpu().numpy()
            if labels_t is None:
                cls_out = np.zeros((len(bboxes), 1), dtype=np.int64)
            else:
                cls_out = labels_t.to(torch.int64).view(-1, 1).cpu().numpy()

        labels["img"] = img_np
        labels["instances"] = Instances(bboxes=bboxes, bbox_format="xyxy", normalized=False)
        labels["cls"] = cls_out
        labels["resized_shape"] = img_np.shape[:2]
        return labels


class _RTDETRRandomIoUCrop:
    def __init__(self, p: float = 1.0, **kwargs) -> None:
        import torchvision.transforms.v2 as T

        self.p = p
        self.transform = T.RandomIoUCrop(**kwargs)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if torch.rand(1) >= self.p:
            return labels
        return self.transform(labels)


def rtdetr_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False):
    """Apply a series of image transformations for RT-DETR training."""
    del dataset, stretch  # Unused, kept for API compatibility with existing transform builders.
    import torchvision.transforms.v2 as T

    if not hasattr(hyp, "fliplr"):
        raise AttributeError("rtdetr_transforms requires 'fliplr' in hyp.")
    fliplr = float(hyp.fliplr)
    return Compose(
        [
            _RTDETRToTvTensors(),
            T.RandomPhotometricDistort(p=0.5),
            T.RandomZoomOut(fill=0),
            _RTDETRRandomIoUCrop(p=0.8),
            T.SanitizeBoundingBoxes(min_size=1),
            T.RandomHorizontalFlip(p=fliplr),
            T.Resize(size=[imgsz, imgsz]),
            T.SanitizeBoundingBoxes(min_size=1),
            _RTDETRFromTvTensors(),
        ]
    )  # transforms


class _RTDETRDEIMPolicy:
    """Epoch-aware DEIM 4-stage transform policy.

    Stages with 0-based epoch indexing:
      1) [0, start): disable policy ops
      2) [start, mid): per-sample branch by mosaic_prob
         - branch A: Mosaic + Photometric
         - branch B: Photometric + ZoomOut + IoUCrop
      3) [mid, stop): Photometric + ZoomOut + IoUCrop
      4) [stop, +inf): disable policy ops
    """

    def __init__(
        self,
        dataset,
        imgsz: int,
        fliplr: float,
        policy_epochs: tuple[int, int, int],
        mosaic_prob: float,
    ) -> None:
        import torchvision.transforms.v2 as T

        self.to_tv = _RTDETRToTvTensors()
        # DEIM Mosaic op itself runs with probability 1.0; branch routing is handled externally via mosaic_prob.
        self.mosaic = _RTDETRDEIMMosaic(dataset, imgsz=imgsz, p=1.0)
        self.photometric = T.RandomPhotometricDistort(p=0.5)
        self.zoomout = T.RandomZoomOut(fill=0)
        self.ioucrop = _RTDETRRandomIoUCrop(p=0.8)
        self.sanitize1 = T.SanitizeBoundingBoxes(min_size=1)
        self.flip = T.RandomHorizontalFlip(p=fliplr)
        self.resize = T.Resize(size=[imgsz, imgsz])
        self.sanitize2 = T.SanitizeBoundingBoxes(min_size=1)
        self.from_tv = _RTDETRFromTvTensors()

        self.policy_epochs = policy_epochs
        self.mosaic_prob = mosaic_prob
        self.epoch = 0
        self.post_transforms = []

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch (0-based) for stage scheduling."""
        self.epoch = epoch

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        start, mid, stop = self.policy_epochs
        cur_epoch = self.epoch

        if start <= cur_epoch < mid:
            with_mosaic = random.random() <= self.mosaic_prob
            labels = self.mosaic(labels) if with_mosaic else self.to_tv(labels)
            labels = self.photometric(labels)
            if not with_mosaic:
                labels = self.zoomout(labels)
                labels = self.ioucrop(labels)
        elif mid <= cur_epoch < stop:
            labels = self.to_tv(labels)
            labels = self.photometric(labels)
            labels = self.zoomout(labels)
            labels = self.ioucrop(labels)
        else:
            # first and last stages: no DEIM policy ops
            labels = self.to_tv(labels)

        # Always-on ops
        labels = self.sanitize1(labels)
        labels = self.flip(labels)
        labels = self.resize(labels)
        labels = self.sanitize2(labels)
        labels = self.from_tv(labels)
        for transform in self.post_transforms:
            labels = transform(labels)
        return labels

    def append(self, transform) -> None:
        """Append post-transform ops (e.g., Format) for API compatibility with Compose-like callers."""
        self.post_transforms.append(transform)


class _RTDETRDEIMMosaic:
    """DEIM-style Mosaic that keeps data in torchvision tv_tensor format."""

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0) -> None:
        import torchvision.transforms.v2 as T
        from torchvision import tv_tensors

        self.dataset = dataset
        self.half_size = imgsz // 2
        self.p = p
        self._tv_tensors = tv_tensors
        self._resize = T.Resize(size=[self.half_size, self.half_size])
        self._affine = T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.5, 1.5), fill=0)
        self._sanitize = T.SanitizeBoundingBoxes(min_size=1)

    def _convert_to_pil(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels.pop("img")
        instances = labels.pop("instances", None)
        cls = labels.pop("cls")

        h, w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 3:
            image = Image.fromarray(img[..., ::-1].copy())  # BGR -> RGB
        else:
            image = Image.fromarray(img.copy())

        if instances is None or len(instances) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            instances.convert_bbox(format="xyxy")
            if instances.segments is None:
                instances.segments = np.zeros((0, 0, 2), dtype=np.float32)
            instances.denormalize(w, h)
            boxes_tensor = torch.as_tensor(instances.bboxes, dtype=torch.float32)

        cls_tensor = (
            torch.as_tensor(cls.reshape(-1), dtype=torch.int64) if len(cls) else torch.zeros((0,), dtype=torch.int64)
        )

        labels["image"] = image
        labels["boxes"] = self._tv_tensors.BoundingBoxes(boxes_tensor, format="XYXY", canvas_size=(h, w))
        labels["labels"] = cls_tensor
        return labels

    def _mosaic4(self, labels_list: list[dict[str, Any]]) -> dict[str, Any]:
        import torchvision.transforms.v2.functional as TF

        s = self.half_size
        canvas_size = s * 2
        canvas = torch.zeros((3, canvas_size, canvas_size), dtype=torch.uint8)
        offsets = [(0, 0), (s, 0), (0, s), (s, s)]

        all_boxes, all_cls = [], []
        for lbl, (x_off, y_off) in zip(labels_list, offsets):
            img_t = TF.pil_to_tensor(lbl["image"])
            h, w = img_t.shape[1], img_t.shape[2]
            ph, pw = min(h, s), min(w, s)
            canvas[:, y_off : y_off + ph, x_off : x_off + pw] = img_t[:, :ph, :pw]

            boxes = lbl["boxes"]
            if len(boxes):
                boxes = boxes.clone()
                boxes[:, [0, 2]] += x_off
                boxes[:, [1, 3]] += y_off
                all_boxes.append(boxes)
                all_cls.append(lbl["labels"])

        if all_boxes:
            final_boxes = torch.cat(all_boxes, dim=0)
            final_cls = torch.cat(all_cls, dim=0)
        else:
            final_boxes = torch.zeros((0, 4), dtype=torch.float32)
            final_cls = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": TF.to_pil_image(canvas),
            "boxes": self._tv_tensors.BoundingBoxes(final_boxes, format="XYXY", canvas_size=(canvas_size, canvas_size)),
            "labels": final_cls,
        }

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        labels = self._convert_to_pil(labels)
        if random.random() > self.p:
            return labels

        labels["image"], labels["boxes"] = self._resize(labels["image"], labels["boxes"])

        sample_indices = random.choices(range(len(self.dataset)), k=3)
        all_labels = [labels]
        for idx in sample_indices:
            other = self.dataset.get_image_and_label(idx)
            other = self._convert_to_pil(other)
            other["image"], other["boxes"] = self._resize(other["image"], other["boxes"])
            all_labels.append(other)

        mosaic_labels = self._mosaic4(all_labels)
        mosaic_labels["image"], mosaic_labels["boxes"] = self._affine(mosaic_labels["image"], mosaic_labels["boxes"])
        mosaic_labels = self._sanitize(mosaic_labels)
        return mosaic_labels


def _compute_policy_epochs(hyp: IterableSimpleNamespace) -> tuple[int, int, int]:
    """Compute DEIM policy boundaries from epochs only."""
    if not hasattr(hyp, "epochs"):
        raise AttributeError("_compute_policy_epochs requires 'epochs' in hyp.")
    epochs = max(1, int(hyp.epochs))
    start = min(4, max(0, epochs - 1))
    stop = epochs
    mid = min(stop, start + stop // 2)
    return start, mid, stop


def rtdetr_deim_transforms(
    dataset,
    imgsz: int,
    hyp: IterableSimpleNamespace,
    policy_epochs: tuple[int, int, int],
    mosaic_prob: float,
    stretch: bool = False,
):
    """Build epoch-aware DEIM transforms for RT-DETR variants."""
    del stretch  # Unused, kept for API compatibility with existing transform builders.
    if not hasattr(hyp, "fliplr"):
        raise AttributeError("rtdetr_deim_transforms requires 'fliplr' in hyp.")
    fliplr = float(hyp.fliplr)
    return _RTDETRDEIMPolicy(
        dataset=dataset,
        imgsz=imgsz,
        fliplr=fliplr,
        policy_epochs=policy_epochs,
        mosaic_prob=float(mosaic_prob),
    )
