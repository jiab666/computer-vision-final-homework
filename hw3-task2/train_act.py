#!/usr/bin/env python
"""ACT Training on CALVIN dataset."""

import argparse
from pathlib import Path
import torch
import swanlab
from torch.optim import AdamW
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
    "actions_is_pad": "action_is_pad",
}


def preprocess_batch(batch, device):
    """Rename, squeeze, normalize and move to device."""
    out = {}
    for dk, pk in KEY_MAP.items():
        if dk in batch:
            t = batch[dk]
            if isinstance(t, (list, tuple)):
                t = torch.tensor(t)
            # Squeeze time dim from delta_timestamps for observations
            if pk in ('observation.image', 'observation.wrist_image', 'observation.state'):
                t = t.squeeze(1)
            out[pk] = t.to(device)

    # ImageNet normalization
    for k in out:
        if k.startswith("observation.image"):
            out[k] = (out[k] - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_roots", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/hw3_outputs/model_b")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--eval_root", type=str, default=None)
    parser.add_argument("--name", type=str, default="act_calvin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_vae", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data: {args.data_roots}")
    print(f"Output: {output_dir}")

    # Feature definitions
    input_features = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 200, 200)),
        "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
    }
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        n_obs_steps=1,
        dim_model=512, n_heads=8, dim_feedforward=3200,
        n_encoder_layers=4, n_decoder_layers=1,
        use_vae=args.use_vae, latent_dim=32, n_vae_encoder_layers=4,
        dropout=0.1, kl_weight=10.0,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        optimizer_lr=args.lr, optimizer_weight_decay=args.weight_decay,
    )

    policy = ACTPolicy(cfg).train().to(device)

    # Get FPS
    metadata = LeRobotDatasetMetadata("calvin", root=args.data_roots[0])
    fps = metadata.fps
    print(f"FPS: {fps}")

    # Delta timestamps (flat dataset names)
    delta = {
        "image": [0.0], "wrist_image": [0.0], "state": [0.0],
        "actions": [i / fps for i in range(args.chunk_size)],
    }

    # Load dataset(s)
    datasets = []
    for root in args.data_roots:
        ds = LeRobotDataset("calvin", root=root, delta_timestamps=delta)
        datasets.append(ds)
        print(f"  {root}: {len(ds)} frames")
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    print(f"Total: {len(dataset)} frames")

    # Eval loader
    eval_loader = None
    if args.eval_root:
        eval_ds = LeRobotDataset("calvin", root=args.eval_root, delta_timestamps=delta)
        eval_loader = torch.utils.data.DataLoader(
            eval_ds, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, pin_memory=True, drop_last=False,
        )
        print(f"Eval: {len(eval_ds)} frames")

    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, drop_last=True,
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Params: {n_params:,}")

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    swanlab.init(
        mode="local", log_dir="./swanlog",
        project="hw3-act-calvin", experiment_name=args.name,
        config={"chunk_size": args.chunk_size, "batch_size": args.batch_size,
                "lr": args.lr, "steps": args.steps, "params": n_params},
    )

    scaler = torch.amp.GradScaler("cuda")
    step = start_step
    total_loss = total_l1 = total_kl = 0.0
    best_eval_loss = float("inf")
    log_int = args.log_freq

    print(f"\nTraining {args.steps} steps\n")

    data_iter = iter(dataloader)
    while step < start_step + args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = preprocess_batch(batch, device)

        with torch.amp.autocast("cuda"):
            loss, loss_dict = policy.forward(batch)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        l1 = loss_dict.get("l1_loss", 0.0)
        kl = loss_dict.get("kld_loss", 0.0)
        if hasattr(l1, "item"): l1 = l1.item()
        if hasattr(kl, "item"): kl = kl.item()

        total_loss += loss.item()
        total_l1 += l1
        total_kl += kl
        step += 1

        if step % log_int == 0:
            avg_loss = total_loss / log_int
            avg_l1 = total_l1 / log_int
            avg_kl = total_kl / log_int
            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | L1: {avg_l1:.4f} | KL: {avg_kl:.4f}")
            swanlab.log({"train/loss": avg_loss, "train/l1_loss": avg_l1,
                         "train/kl_loss": avg_kl, "train/step": step})
            total_loss = total_l1 = total_kl = 0.0

        # Eval
        if eval_loader and step % args.eval_freq == 0:
            policy.eval()
            e_loss = e_l1 = 0.0
            e_count = 0
            with torch.no_grad():
                for eb in eval_loader:
                    eb = preprocess_batch(eb, device)
                    with torch.amp.autocast("cuda"):
                        el, ed = policy.forward(eb)
                    el1 = ed.get("l1_loss", 0.0)
                    if hasattr(el1, "item"): el1 = el1.item()
                    e_loss += el.item()
                    e_l1 += el1
                    e_count += 1
                    if e_count >= 100:
                        break
            avg_el = e_loss / e_count
            avg_el1 = e_l1 / e_count
            print(f"Eval {step:6d} | Loss: {avg_el:.4f} | L1: {avg_el1:.4f}")
            swanlab.log({"eval/loss": avg_el, "eval/l1_loss": avg_el1, "eval/step": step})
            if avg_el < best_eval_loss:
                best_eval_loss = avg_el
                torch.save({"step": step, "model_state_dict": policy.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "loss": avg_el},
                           output_dir / "best_model.pt")
            policy.train()

        if step % args.save_freq == 0:
            torch.save({"step": step, "model_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict()},
                       output_dir / f"checkpoint_{step:06d}.pt")

    print(f"\nDone! Final step: {step}")
    policy.save_pretrained(output_dir / "final_model")
    print(f"Saved to {output_dir / 'final_model'}")
    swanlab.finish()


if __name__ == "__main__":
    main()
