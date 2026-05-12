"""
Optional pretrained CNN baseline helpers.
The current implementation expects local TorchScript weights.
"""
from __future__ import annotations

import os

import numpy as np


MODEL_SPECS = [
    {
        "id": "zerodcepp",
        "display_name": "Zero-DCE++",
        "env_var": "ZERO_DCEPP_MODEL_PATH",
        "candidate_paths": [
            os.path.join("models", "zerodcepp", "zerodcepp.ts"),
            os.path.join("models", "zerodcepp", "model.ts"),
            os.path.join("models", "zerodcepp", "zerodcepp.pt"),
            os.path.join("models", "zerodcepp", "model.pt"),
        ],
    },
    {
        "id": "retinexnet",
        "display_name": "RetinexNet",
        "env_var": "RETINEXNET_MODEL_PATH",
        "candidate_paths": [
            os.path.join("models", "retinexnet", "retinexnet.ts"),
            os.path.join("models", "retinexnet", "model.ts"),
            os.path.join("models", "retinexnet", "retinexnet.pt"),
            os.path.join("models", "retinexnet", "model.pt"),
        ],
    },
]


def describe_expected_model_locations():
    lines = ["Expected pretrained TorchScript locations:"]
    for model_spec in MODEL_SPECS:
        lines.append(
            f"- {model_spec['display_name']}: env {model_spec['env_var']} or "
            + ", ".join(model_spec["candidate_paths"])
        )
    return "\n".join(lines)


def _resolve_model_path(model_spec):
    configured_path = os.environ.get(model_spec["env_var"])
    candidate_paths = []
    if configured_path:
        candidate_paths.append(configured_path)
    candidate_paths.extend(model_spec["candidate_paths"])

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def find_available_model_spec():
    for model_spec in MODEL_SPECS:
        resolved_path = _resolve_model_path(model_spec)
        if resolved_path is not None:
            resolved = dict(model_spec)
            resolved["path"] = resolved_path
            return resolved
    return None


def find_all_available_model_specs():
    resolved_models = []
    for model_spec in MODEL_SPECS:
        resolved_path = _resolve_model_path(model_spec)
        if resolved_path is not None:
            resolved = dict(model_spec)
            resolved["path"] = resolved_path
            resolved_models.append(resolved)
    return resolved_models


def _extract_tensor(output):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CNN comparison branch.") from exc

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def load_torchscript_model(model_path, device="cpu"):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CNN comparison branch.") from exc

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def run_torchscript_model(model, rgb_uint8, device="cpu"):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CNN comparison branch.") from exc

    input_tensor = torch.from_numpy(np.transpose(rgb_uint8, (2, 0, 1))).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output_tensor = _extract_tensor(output)
    if output_tensor is None:
        raise RuntimeError("Could not find an image tensor in the model output.")

    if output_tensor.ndim == 3:
        output_tensor = output_tensor.unsqueeze(0)

    if output_tensor.shape[1] == 1:
        output_tensor = output_tensor.repeat(1, 3, 1, 1)

    if output_tensor.shape[-2:] != input_tensor.shape[-2:]:
        output_tensor = F.interpolate(
            output_tensor,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    output_tensor = output_tensor[0].detach().cpu().clamp(0, 1)
    output_rgb = np.transpose(output_tensor.numpy(), (1, 2, 0))
    return np.clip(255.0 * output_rgb, 0, 255).astype(np.uint8)
