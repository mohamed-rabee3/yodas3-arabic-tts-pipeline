"""DiariZen (BUT-FIT, WavLM-Large-s80 + Conformer, powerset EEND) local segmentation model,
loaded without DiariZen's pinned pyannote-3.1 fork.

Model weights: BUT-FIT/diarizen-wavlm-large-s80-md-v2 — CC BY-NC 4.0 (non-commercial).
Code: github.com/BUTSpeechFIT/DiariZen (MIT), cloned at /workspace/DiariZen.
"""
import sys
import types

import toml
import torch
import torch.nn as nn

DIARIZEN_SRC = "/workspace/DiariZen"
CKPT_DIR = "/workspace/models/diarizen/large-s80-v2"


class _Base(nn.Module):
    """Minimal stand-in for pyannote-3.1 Model: just specifications + default activation."""

    def __init__(self, num_channels=1, duration=16, max_speakers_per_chunk=4, max_speakers_per_frame=2):
        super().__init__()
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        self.specifications = Specifications(
            problem=Problem.MONO_LABEL_CLASSIFICATION, resolution=Resolution.FRAME,
            duration=duration, min_duration=duration, warm_up=(0.0, 0.0),
            classes=[f"speaker#{i + 1}" for i in range(max_speakers_per_chunk)],
            powerset_max_classes=max_speakers_per_frame, permutation_invariant=True)

    def default_activation(self):
        return nn.LogSoftmax(dim=-1)


def load_diarizen(device="cuda"):
    if DIARIZEN_SRC not in sys.path:
        sys.path.insert(0, DIARIZEN_SRC)
    # model_wavlm_conformer imports `pyannote.audio.core.model.Model`; give it our shim instead
    import pyannote.audio.core.model as pam
    real = pam.Model
    pam.Model = _Base
    try:
        import importlib
        mod = importlib.import_module("diarizen.models.eend.model_wavlm_conformer")
        mod = importlib.reload(mod)
    finally:
        pam.Model = real
    cfg = toml.load(f"{CKPT_DIR}/config.toml")["model"]["args"]
    model = mod.Model(**cfg)
    sd = torch.load(f"{CKPT_DIR}/pytorch_model.bin", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "dummy" not in k]
    if missing or unexpected:
        raise RuntimeError(f"DiariZen state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    from pyannote.audio.utils.powerset import Powerset
    ps = Powerset(cfg["max_speakers_per_chunk"], cfg["max_speakers_per_frame"]).to(device)
    return model, ps, cfg["chunk_size"]
