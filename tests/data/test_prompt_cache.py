"""Prompt-cache integration tests for the Wan camera LMDB dataset."""

import hashlib
import json

import lmdb
import numpy as np
import pytest
import torch

from minwm.data.datasets.lmdb import CameraLatentLMDBDataset
from minwm.modeling.wan21.adapter import Wan21Adapter


def _write_source(path, prompt="a moving camera"):
    env = lmdb.open(str(path), map_size=4 * 1024**2)
    with env.begin(write=True) as txn:
        for key, shape in (("latents", (1, 2, 1, 4, 4)), ("intrinsics", (1, 4)), ("poses", (1, 2, 7))):
            txn.put(f"{key}_shape".encode(), " ".join(map(str, shape)).encode())
        txn.put(b"latents_0_data", np.zeros((2, 1, 4, 4), dtype=np.float16).tobytes())
        txn.put(b"prompts_0_data", prompt.encode())
        txn.put(b"intrinsics_0_data", np.array([[1, 1, 0.5, 0.5]], dtype=np.float32).tobytes())
        poses = np.zeros((2, 7), dtype=np.float32)
        poses[:, 6] = 1
        txn.put(b"poses_0_data", poses.tobytes())
    env.close()


def _write_cache(path, prompt="a moving camera", context=None):
    context = context if context is not None else torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
    env = lmdb.open(str(path), map_size=4 * 1024**2)
    with env.begin(write=True) as txn:
        txn.put(
            b"metadata",
            json.dumps({
                "format": "wan21_prompt_context_v1",
                "num_samples": 1,
                "embedding_dtype": "bfloat16",
            }).encode(),
        )
        txn.put(b"context_0_data", context.view(torch.uint16).numpy().tobytes())
        txn.put(b"context_0_shape", "3 4".encode())
        txn.put(b"prompt_0_sha256", hashlib.sha256(prompt.encode()).hexdigest().encode())
    env.close()
    return context


def test_camera_dataset_loads_context_and_adapter_skips_encoder(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    _write_source(source)
    expected = _write_cache(cache)
    sample = CameraLatentLMDBDataset(str(source), prompt_cache_path=str(cache))[0]
    assert torch.equal(sample["prompt_context"], expected)

    class FailingEncoder(torch.nn.Module):
        def forward(self, prompts):
            raise AssertionError("cached prompt embeddings should bypass UMT5")

    adapter = Wan21Adapter(text_dim=4, dtype="bfloat16", text_encoder=FailingEncoder())
    context = adapter.conditioning({"prompt_context": [sample["prompt_context"]]}, 1, torch.device("cpu"))[
        "context"
    ]
    assert torch.equal(context[0], expected)


def test_camera_dataset_rejects_prompt_mismatch(tmp_path):
    source = tmp_path / "source"
    cache = tmp_path / "cache"
    _write_source(source, prompt="changed prompt")
    _write_cache(cache, prompt="original prompt")
    ds = CameraLatentLMDBDataset(str(source), prompt_cache_path=str(cache))
    with pytest.raises(ValueError, match="does not match"):
        ds[0]
