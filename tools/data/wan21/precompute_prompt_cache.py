"""Precompute Wan UMT5 prompt embeddings into an LMDB cache.

Example (single GPU)::

    python -m tools.data.wan21.precompute_prompt_cache \
        --data-lmdb /data/train_lmdb --output-lmdb /data/train_prompt_cache \
        --checkpoint /models/models_t5_umt5-xxl-enc-bf16.pth \
        --tokenizer /models/google/umt5-xxl --batch-size 8

Run once for each fixed dataset split. Samples remain keyed by their original
LMDB index, and source-prompt hashes are checked by CameraLatentLMDBDataset.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import lmdb
import torch

from minwm.data.datasets.lmdb import _get_row, _get_shape, _is_single_lmdb, _open, _open_shards
from minwm.modeling.wan21.text_encoder import Wan21TextEncoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-lmdb", required=True, help="fixed source dataset LMDB")
    p.add_argument("--output-lmdb", required=True, help="new prompt-cache LMDB directory")
    p.add_argument("--checkpoint", required=True, help="Wan UMT5 checkpoint")
    p.add_argument("--tokenizer", required=True, help="UMT5 tokenizer directory")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--text-len", type=int, default=512)
    p.add_argument("--map-size-gb", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_lmdb)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output LMDB already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    if _is_single_lmdb(args.data_lmdb):
        source_env = _open(args.data_lmdb)
        num_samples = _get_shape(source_env, "latents")[0]

        def get_prompt(idx):
            return _get_row(source_env, "prompts", str, idx)

    else:
        source_envs = _open_shards(args.data_lmdb)
        shard_offsets = []
        num_samples = 0
        for env in source_envs:
            count = _get_shape(env, "latents")[0]
            shard_offsets.append((num_samples, num_samples + count))
            num_samples += count

        def get_prompt(idx):
            for sid, (start, stop) in enumerate(shard_offsets):
                if start <= idx < stop:
                    return _get_row(source_envs[sid], "prompts", str, idx - start)
            raise IndexError(idx)

    encoder = Wan21TextEncoder(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        text_len=args.text_len,
        dtype="bfloat16",
    ).to(args.device).eval()
    env = lmdb.open(str(output), map_size=args.map_size_gb * 1024**3, subdir=True)
    metadata = {
        "format": "wan21_prompt_context_v1",
        "num_samples": num_samples,
        "text_len": args.text_len,
        "embedding_dtype": "bfloat16",
        "text_dim": 4096,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_size": os.path.getsize(args.checkpoint),
        "tokenizer": str(Path(args.tokenizer).resolve()),
    }
    with env.begin(write=True) as txn:
        txn.put(b"metadata", json.dumps(metadata, sort_keys=True).encode())

    with torch.inference_mode():
        for start in range(0, num_samples, args.batch_size):
            stop = min(start + args.batch_size, num_samples)
            prompts = [get_prompt(i) for i in range(start, stop)]
            contexts = encoder(prompts)
            with env.begin(write=True) as txn:
                for idx, prompt, context in zip(range(start, stop), prompts, contexts):
                    context = context.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    raw = context.view(torch.uint16).numpy().tobytes()
                    prefix = f"context_{idx}".encode()
                    txn.put(prefix + b"_data", raw)
                    txn.put(prefix + b"_shape", " ".join(map(str, context.shape)).encode())
                    txn.put(
                        f"prompt_{idx}_sha256".encode(),
                        hashlib.sha256(prompt.encode("utf-8")).hexdigest().encode(),
                    )
            if stop == num_samples or stop % (args.batch_size * 100) == 0:
                print(f"encoded {stop}/{num_samples} prompts", flush=True)
    env.sync()
    env.close()
    print(f"prompt cache saved to {output}")


if __name__ == "__main__":
    main()
