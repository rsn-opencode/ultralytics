#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for stereo3ddet config validation and model forward pass."""

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.stereo3ddet.head_yolo11 import Stereo3DDetHeadYOLO11
from ultralytics.nn.modules.head import Detect


class TestStereo3DDetConfigHead:
    """Test that head uses Stereo3DDetHeadYOLO11 (extends Detect)."""

    def test_yolo11_head_is_stereo3ddet(self):
        """YOLO11 stereo config uses Stereo3DDetHeadYOLO11 as last layer."""
        model = YOLO("yolo11n-stereo3ddet.yaml")
        head = model.model.model[-1]
        assert isinstance(head, Stereo3DDetHeadYOLO11)
        assert isinstance(head, Detect)  # subclass of Detect
        assert model.task == "stereo3ddet"

    def test_yolo26_head_is_stereo3ddet(self):
        """YOLO26 stereo config uses same Stereo3DDetHeadYOLO11 as last layer."""
        model = YOLO("yolo26n-stereo3ddet.yaml")
        head = model.model.model[-1]
        assert isinstance(head, Stereo3DDetHeadYOLO11)
        assert isinstance(head, Detect)
        assert model.task == "stereo3ddet"


class TestStereo3DDetModelInstantiation:
    """Test model instantiation and properties."""

    def test_end2end_forced_off(self):
        """Stereo model forces end2end=False regardless of config."""
        for cfg in ("yolo11n-stereo3ddet.yaml", "yolo26n-stereo3ddet.yaml"):
            model = YOLO(cfg)
            assert model.model.end2end is False, f"{cfg}: end2end should be False"

    def test_reg_max_is_one(self):
        """Stereo head forces reg_max=1 (no DFL)."""
        model = YOLO("yolo11n-stereo3ddet.yaml")
        head = model.model.model[-1]
        assert head.reg_max == 1

    def test_aux_branches_exist(self):
        """Head has all expected aux branches."""
        model = YOLO("yolo11n-stereo3ddet.yaml")
        head = model.model.model[-1]
        expected = {"lr_distance", "dimensions", "orientation", "depth"}
        assert set(head.aux.keys()) == expected


class TestStereo3DDetForwardPass:
    """Test model forward pass produces correct output format."""

    def test_training_output_format(self):
        """Training forward returns dict with boxes, scores, feats, and aux branches."""
        model = YOLO("yolo11n-stereo3ddet.yaml")
        model.model.train()
        x = torch.randn(1, 6, 384, 1280)
        out = model.model(x)

        assert isinstance(out, dict)
        assert "boxes" in out
        assert "scores" in out
        assert "feats" in out
        assert "lr_distance" in out
        assert "dimensions" in out
        assert "orientation" in out
        assert "depth" in out

    def test_eval_output_format(self):
        """Eval forward returns (y, preds_dict) tuple."""
        model = YOLO("yolo11n-stereo3ddet.yaml")
        model.model.eval()
        x = torch.randn(1, 6, 384, 1280)
        with torch.no_grad():
            out = model.model(x)

        assert isinstance(out, tuple) and len(out) == 2
        y, preds = out
        assert isinstance(y, torch.Tensor)  # inference output
        assert isinstance(preds, dict)
        assert "lr_distance" in preds
        assert "depth" in preds

    def test_aux_output_shapes(self):
        """Aux branch outputs have correct [B, C, HW_total] shape."""
        model = YOLO("yolo11n-stereo3ddet.yaml")
        model.model.train()
        x = torch.randn(2, 6, 384, 1280)
        out = model.model(x)

        assert out["lr_distance"].shape[0] == 2  # batch
        assert out["lr_distance"].shape[1] == 1  # 1 channel
        assert out["dimensions"].shape[1] == 3  # 3 channels (H, W, L)
        assert out["orientation"].shape[1] == 2  # sin, cos
        assert out["depth"].shape[1] == 1  # 1 channel
        # All aux branches should have same HW_total
        hw = out["lr_distance"].shape[2]
        assert out["dimensions"].shape[2] == hw
        assert out["orientation"].shape[2] == hw
        assert out["depth"].shape[2] == hw

    def test_yolo26_forward_matches_yolo11(self):
        """YOLO26 stereo produces same output keys as YOLO11."""
        m11 = YOLO("yolo11n-stereo3ddet.yaml")
        m26 = YOLO("yolo26n-stereo3ddet.yaml")
        m11.model.train()
        m26.model.train()
        x = torch.randn(1, 6, 384, 1280)
        out11 = m11.model(x)
        out26 = m26.model(x)
        assert set(out11.keys()) == set(out26.keys())
