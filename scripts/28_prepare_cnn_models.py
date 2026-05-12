"""
Script 28: Convert raw .pth/.tar weights of Zero-DCE++ and RetinexNet into
TorchScript .ts files that utils/cnn_baseline.py can load directly.

Each model is wrapped in a thin nn.Module whose forward(rgb_01) returns the
final enhanced RGB tensor in [0, 1] only (no intermediate tuples).

Expected weights at:
- models/zerodcepp/Epoch99.pth          (Zero-DCE++)
- models/retinexnet/Decom_9200.tar      (RetinexNet DecomNet)
- models/retinexnet/Relight_9200.tar    (RetinexNet RelightNet)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.external.zerodcepp.model import enhance_net_nopool
from utils.external.retinexnet.model import DecomNet, RelightNet


ZERODCEPP_WEIGHTS = os.path.join("models", "zerodcepp", "Epoch99.pth")
ZERODCEPP_TORCHSCRIPT = os.path.join("models", "zerodcepp", "zerodcepp.ts")
ZERODCEPP_SCALE_FACTOR = 12
RETINEXNET_DECOM_WEIGHTS = os.path.join("models", "retinexnet", "Decom_9200.tar")
RETINEXNET_RELIGHT_WEIGHTS = os.path.join("models", "retinexnet", "Relight_9200.tar")
RETINEXNET_TORCHSCRIPT = os.path.join("models", "retinexnet", "retinexnet.ts")


class ZeroDcePpWrapper(nn.Module):
    __constants__ = ["scale_factor"]

    def __init__(self, net, scale_factor):
        super().__init__()
        self.net = net
        self.scale_factor = scale_factor

    def forward(self, x):
        height = x.shape[2]
        width = x.shape[3]
        height_padding = (self.scale_factor - height % self.scale_factor) % self.scale_factor
        width_padding = (self.scale_factor - width % self.scale_factor) % self.scale_factor
        padded = F.pad(x, [0, width_padding, 0, height_padding], mode="reflect")
        enhance_image, _ = self.net(padded)
        cropped = enhance_image[:, :, :height, :width]
        return torch.clamp(cropped, 0.0, 1.0)


class RetinexNetWrapper(nn.Module):
    def __init__(self, decom_net, relight_net):
        super().__init__()
        self.decom_net = decom_net
        self.relight_net = relight_net

    def forward(self, x):
        R_low, I_low = self.decom_net(x)
        I_delta = self.relight_net(I_low, R_low)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)
        output = R_low * I_delta_3
        return torch.clamp(output, 0.0, 1.0)


def strip_data_parallel_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[len("module."):] if key.startswith("module.") else key] = value
    return cleaned


def build_zerodcepp():
    if not os.path.exists(ZERODCEPP_WEIGHTS):
        raise FileNotFoundError(
            f"Zero-DCE++ weights not found at {ZERODCEPP_WEIGHTS}."
        )
    net = enhance_net_nopool(scale_factor=ZERODCEPP_SCALE_FACTOR)
    state_dict = torch.load(ZERODCEPP_WEIGHTS, map_location="cpu", weights_only=True)
    net.load_state_dict(strip_data_parallel_prefix(state_dict))
    net.eval()
    return ZeroDcePpWrapper(net, ZERODCEPP_SCALE_FACTOR).eval()


def build_retinexnet():
    if not os.path.exists(RETINEXNET_DECOM_WEIGHTS):
        raise FileNotFoundError(
            f"RetinexNet DecomNet weights not found at {RETINEXNET_DECOM_WEIGHTS}."
        )
    if not os.path.exists(RETINEXNET_RELIGHT_WEIGHTS):
        raise FileNotFoundError(
            f"RetinexNet RelightNet weights not found at {RETINEXNET_RELIGHT_WEIGHTS}."
        )

    decom_net = DecomNet()
    relight_net = RelightNet()

    decom_state = torch.load(RETINEXNET_DECOM_WEIGHTS, map_location="cpu", weights_only=True)
    relight_state = torch.load(RETINEXNET_RELIGHT_WEIGHTS, map_location="cpu", weights_only=True)

    decom_net.load_state_dict(strip_data_parallel_prefix(decom_state))
    relight_net.load_state_dict(strip_data_parallel_prefix(relight_state))

    decom_net.eval()
    relight_net.eval()
    return RetinexNetWrapper(decom_net, relight_net).eval()


def export_module(module, output_path, dummy_input):
    with torch.no_grad():
        try:
            scripted = torch.jit.script(module)
        except Exception as script_error:
            print(f"  scripting failed: {type(script_error).__name__}: {script_error}")
            print("  falling back to tracing")
            scripted = torch.jit.trace(module, dummy_input)
        scripted.save(output_path)


def smoke_test(output_path, dummy_input):
    with torch.no_grad():
        loaded = torch.jit.load(output_path, map_location="cpu")
        loaded.eval()
        result = loaded(dummy_input)
    assert isinstance(result, torch.Tensor), "Loaded model did not return a Tensor"
    assert result.dim() == 4 and result.shape[1] == 3, f"Unexpected output shape: {tuple(result.shape)}"
    assert torch.isfinite(result).all(), "Output contains non-finite values"
    minimum = float(result.min())
    maximum = float(result.max())
    print(f"  loaded OK -> shape {tuple(result.shape)}, range [{minimum:.4f}, {maximum:.4f}]")


if __name__ == "__main__":
    dummy = torch.rand(1, 3, 256, 384)

    print("Building Zero-DCE++ wrapper...")
    zero_dce = build_zerodcepp()
    with torch.no_grad():
        sample_out = zero_dce(dummy)
    print(f"  forward shape: {tuple(sample_out.shape)}")
    export_module(zero_dce, ZERODCEPP_TORCHSCRIPT, dummy)
    print(f"  saved -> {ZERODCEPP_TORCHSCRIPT}")
    smoke_test(ZERODCEPP_TORCHSCRIPT, dummy)

    print("Building RetinexNet wrapper...")
    retinex_net = build_retinexnet()
    with torch.no_grad():
        sample_out = retinex_net(dummy)
    print(f"  forward shape: {tuple(sample_out.shape)}")
    export_module(retinex_net, RETINEXNET_TORCHSCRIPT, dummy)
    print(f"  saved -> {RETINEXNET_TORCHSCRIPT}")
    smoke_test(RETINEXNET_TORCHSCRIPT, dummy)

    print("Done. TorchScript modules ready for script 26.")
