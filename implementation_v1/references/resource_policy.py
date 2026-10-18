"""Local resource policy for complete plaintext reference models.

The user has prohibited CPU large-model inference. Small independent integer
operators remain usable. Complete R0/R1 models must use the actual RTX 2060
CUDA backend; NumPy on the host is not such a backend.
"""


def require_model_device(profile, requested_device=None):
    if profile not in ("R0","R1"):
        raise ValueError("UNKNOWN_REFERENCE_PROFILE")
    import torch
    device = torch.device(requested_device or "cuda")
    if device.type != "cuda":
        raise RuntimeError("REFERENCE_CPU_MODEL_INFERENCE_DISABLED")
    if not torch.cuda.is_available():
        raise RuntimeError(profile+"_CUDA_UNAVAILABLE_CPU_FALLBACK_PROHIBITED")
    try:
        if requested_device is None:
            matches=[i for i in range(torch.cuda.device_count()) if '2060' in torch.cuda.get_device_name(i)]
            if len(matches)!=1:raise RuntimeError("REQUIRED_EXACTLY_ONE_RTX_2060")
            device=torch.device('cuda',matches[0])
        if '2060' not in torch.cuda.get_device_name(device):raise RuntimeError("REQUIRED_RTX_2060_NOT_SELECTED")
        # Verify actual allocation before reading any checkpoint or running a model.
        torch.empty(1, device=device)
    except (RuntimeError, AssertionError) as error:
        raise RuntimeError(profile+"_CUDA_UNUSABLE_CPU_FALLBACK_PROHIBITED") from error
    return device
