#!/usr/bin/env python
"""Evaluate ACT on test split - computes action prediction L1 loss."""

import argparse, json
from pathlib import Path
import torch
import numpy as np
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.configs import FeatureType, PolicyFeature

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

KEY_MAP = {
    "image": "observation.image",
    "wrist_image": "observation.wrist_image",
    "state": "observation.state",
    "actions": "action",
}

SQ_KEYS = ('observation.image', 'observation.wrist_image', 'observation.state')


def preprocess_batch(batch, device):
    out = {}
    for dk, pk in KEY_MAP.items():
        if dk in batch:
            t = batch[dk]
            if isinstance(t, (list, tuple)):
                t = torch.tensor(t)
            if pk in SQ_KEYS:
                t = t.squeeze(1)
            out[pk] = t.to(device)
    for k in out:
        if k.startswith("observation.image"):
            out[k] = (out[k] - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    model_path = Path(args.model_path)

    metadata = LeRobotDatasetMetadata("calvin", root=args.data_root)
    fps = metadata.fps

    input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 200, 200)),
        "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
    }
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

    cfg = ACTConfig(
        input_features=input_features, output_features=output_features,
        chunk_size=args.chunk_size, n_action_steps=args.chunk_size,
        n_obs_steps=1, use_vae=True,
    )

    policy = ACTPolicy(cfg)
    ckpt = torch.load(str(model_path), map_location=device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device)
    policy.eval()

    delta = {
        "image": [0.0], "wrist_image": [0.0], "state": [0.0],
        "actions": [i / fps for i in range(args.chunk_size)],
    }
    dataset = LeRobotDataset("calvin", root=args.data_root, delta_timestamps=delta)
    print(f"Test dataset: {len(dataset)} frames")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=False,
    )

    total_l1 = 0.0
    per_dim_l1 = np.zeros(7)
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = preprocess_batch(batch, device)
            
            # Use predict_action_chunk for clean inference
            pred_actions = policy.predict_action_chunk(batch)
            target = batch["action"]
            
            l1 = torch.nn.functional.l1_loss(pred_actions, target)
            
            total_l1 += l1.item()
            per_dim_l1 += (pred_actions - target).abs().mean(dim=(0, 1)).cpu().numpy()
            count += 1

            if count % 200 == 0:
                print(f"  Batch {count} | L1: {total_l1/count:.6f}")
            
            if args.max_batches > 0 and count >= args.max_batches:
                break

    avg_l1 = total_l1 / count
    per_dim_l1 /= count

    print(f"\n{'='*50}")
    print(f"Test Results on {args.data_root}")
    print(f"{'='*50}")
    print(f"Batches: {count}")
    print(f"Avg L1:  {avg_l1:.6f}")
    dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    print(f"Per-dim L1:")
    for i, d in enumerate(dims):
        print(f"  {d}: {per_dim_l1[i]:.6f}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "avg_l1": float(avg_l1),
            "per_dim_l1": {d: float(per_dim_l1[i]) for i, d in enumerate(dims)},
            "num_batches": count, "data_root": args.data_root,
            "model_path": str(args.model_path),
        }
        with open(out_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {out_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
