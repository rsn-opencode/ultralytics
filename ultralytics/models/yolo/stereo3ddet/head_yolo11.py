from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


AUX_SPECS = {"lr_distance": 1, "dimensions": 3, "orientation": 2, "depth": 1}


def get_aux_specs(depth_mode: str = "both") -> dict[str, int]:
    """Return filtered AUX_SPECS based on depth_mode.

    Args:
        depth_mode: "both" (default), "lr_only", or "depth_only".
    """
    specs = dict(AUX_SPECS)
    if depth_mode == "lr_only":
        specs.pop("depth", None)
    elif depth_mode == "depth_only":
        specs.pop("lr_distance", None)
    elif depth_mode != "both":
        raise ValueError(f"Unknown depth_mode: {depth_mode!r}. Expected 'both', 'lr_only', or 'depth_only'.")
    return specs


def _branch(in_ch: int, out_ch: int, hidden: int = 256) -> nn.Sequential:
    """Simple conv branch for dense per-location prediction."""
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, 1, 1),
        nn.BatchNorm2d(hidden),
        nn.SiLU(inplace=True),
        nn.Conv2d(hidden, out_ch, 1, 1, 0),
    )


def _deep_branch(in_ch: int, out_ch: int, hidden: int = 64) -> nn.Sequential:
    """Deeper conv branch for depth-critical predictions (lr_distance, depth)."""
    return nn.Sequential(
        Conv(in_ch, hidden, 3),
        Conv(hidden, hidden, 3),
        Conv(hidden, hidden, 3),
        nn.Conv2d(hidden, out_ch, 1),
    )


class Stereo3DDetHeadYOLO11(Detect):
    """Multi-scale stereo 3D detection head (Pose-pattern).

    Receives P3/P4/P5 feature maps from FPN+PAN neck.  Per-scale aux branches
    predict stereo/3D quantities; outputs are flattened to [B, C, HW_total].

    Args:
        nc: Number of classes.
        reg_max: DFL channels (forced to 1).
        end2end: End-to-end mode (forced to False).
        ch: Tuple of per-scale input channels, e.g. (256, 512, 1024).
    """

    def __init__(self, nc: int = 3, reg_max: int = 1, end2end: bool = False, ch: tuple = ()):
        if isinstance(reg_max, (list, tuple)):  # YAML [nc] — ch landed in reg_max slot
            ch, reg_max = reg_max, 1
        super().__init__(nc=nc, reg_max=1, end2end=False, ch=ch)  # Force reg_max=1, end2end=False

        # Force reg_max=1 (no DFL) — stereo 3D detection doesn't benefit from DFL
        self.reg_max = 1
        self.no = nc + 4  # 4 direct bbox offsets, no distribution
        c2 = max(16, ch[0] // 4, 4)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1)) for x in ch
        )
        self.dfl = nn.Identity()

        self.aux_specs = dict(AUX_SPECS)  # mutable copy

        # Hidden size scales with model width (same pattern as Pose.cv4)
        hidden = max(ch[0] // 4, max(self.aux_specs.values()))
        depth_hidden = max(ch[0] // 2, 64)  # wider hidden for depth-critical branches

        # Per-scale aux branches (like Pose.cv4)
        self.aux = nn.ModuleDict()
        for name, out_c in self.aux_specs.items():
            if name in ("lr_distance", "depth"):
                self.aux[name] = nn.ModuleList(_deep_branch(x, out_c, depth_hidden) for x in ch)
            else:
                self.aux[name] = nn.ModuleList(_branch(x, out_c, hidden) for x in ch)

    def set_depth_mode(self, mode: str) -> None:
        """Prune aux branches to match depth_mode ('both', 'lr_only', 'depth_only')."""
        self.aux_specs = get_aux_specs(mode)
        for name in list(self.aux.keys()):
            if name not in self.aux_specs:
                del self.aux[name]

    @property
    def one2many(self):
        """Returns the one-to-many head components including aux branches."""
        return dict(box_head=self.cv2, cls_head=self.cv3, aux_branches=self.aux)

    def forward_head(
        self, x: list[torch.Tensor], box_head=None, cls_head=None, aux_branches=None
    ) -> dict[str, torch.Tensor]:
        """Forward pass: compute detection + aux predictions.

        Returns dict with boxes, scores, feats, and all aux branch outputs.
        """
        preds = super().forward_head(x, box_head, cls_head)  # {boxes, scores, feats}
        if aux_branches is not None:
            bs = x[0].shape[0]
            for name, branches in aux_branches.items():
                out_c = self.aux_specs[name]
                preds[name] = torch.cat(
                    [branches[i](x[i]).view(bs, out_c, -1) for i in range(self.nl)], -1
                )  # [B, C, HW_total]
        return preds
