from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


REPO_DIR = Path(__file__).resolve().parents[1] / "2d-gaussian-splatting"
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR))

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = torch.clamp(tensor.detach().cpu(), 0.0, 1.0)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def camera_to_metadata(camera) -> dict:
    width = int(camera.image_width)
    height = int(camera.image_height)
    ndc2pix = torch.tensor(
        [
            [width / 2, 0, 0, (width - 1) / 2],
            [0, height / 2, 0, (height - 1) / 2],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=camera.projection_matrix.device,
    ).T
    intrinsic = (camera.projection_matrix @ ndc2pix)[:3, :3].T.detach().cpu().numpy()
    extrinsic = camera.world_view_transform.T.detach().cpu().numpy()
    return {
        "width": width,
        "height": height,
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "extrinsic": extrinsic.tolist(),
    }


def foreground_mask_grabcut(image: np.ndarray, iterations: int = 4) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --mask grabcut.")

    height, width = image.shape[:2]
    margin_x = max(4, round(width * 0.08))
    margin_top = max(4, round(height * 0.03))
    margin_bottom = max(4, round(height * 0.02))
    rect = (margin_x, margin_top, width - 2 * margin_x, height - margin_top - margin_bottom)

    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RGB/depth inputs for Open3D TSDF fusion from a 2DGS model.")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=7000, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha-threshold", default=0.05, type=float)
    parser.add_argument("--mask", choices=["none", "grabcut"], default="none")
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    output_dir = Path(args.output)
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    mask_dir = output_dir / "mask"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    if args.mask != "none":
        mask_dir.mkdir(parents=True, exist_ok=True)

    cameras = scene.getTrainCameras()[:: max(1, args.stride)]
    if args.max_images > 0:
        cameras = cameras[: args.max_images]

    frames = []
    all_valid_depths = []
    for idx, camera in enumerate(cameras):
        package = render(camera, gaussians, pipe, background)
        color = tensor_to_uint8(package["render"])
        gt_color = tensor_to_uint8(camera.original_image)
        depth = package["surf_depth"][0].detach().cpu().numpy().astype(np.float32)
        alpha = package.get("rend_alpha")
        if alpha is not None:
            alpha_np = alpha[0].detach().cpu().numpy()
            depth[alpha_np < args.alpha_threshold] = 0.0

        mask_path = None
        if args.mask == "grabcut":
            foreground = foreground_mask_grabcut(gt_color)
            depth[~foreground] = 0.0
            mask_path = mask_dir / f"{idx:05d}.png"
            Image.fromarray((foreground.astype(np.uint8) * 255)).save(mask_path)

        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0

        name = f"{idx:05d}"
        color_path = color_dir / f"{name}.png"
        depth_path = depth_dir / f"{name}.npy"
        Image.fromarray(color).save(color_path)
        np.save(depth_path, depth)

        valid = depth[depth > 0.0]
        if valid.size:
            all_valid_depths.append(valid)

        meta = camera_to_metadata(camera)
        meta.update(
            {
                "name": name,
                "source_camera": camera.image_name,
                "color": color_path.relative_to(output_dir).as_posix(),
                "depth": depth_path.relative_to(output_dir).as_posix(),
                "mask": mask_path.relative_to(output_dir).as_posix() if mask_path else None,
            }
        )
        frames.append(meta)

    if all_valid_depths:
        depths = np.concatenate(all_valid_depths)
        depth_trunc = float(np.percentile(depths, 98.0))
        depth_min = float(np.percentile(depths, 1.0))
        depth_max = float(np.percentile(depths, 99.5))
    else:
        depth_trunc = 5.0
        depth_min = 0.0
        depth_max = 5.0

    metadata = {
        "iteration": scene.loaded_iter,
        "num_frames": len(frames),
        "alpha_threshold": args.alpha_threshold,
        "mask": args.mask,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_trunc_suggestion": depth_trunc,
        "frames": frames,
    }
    (output_dir / "transforms_tsdf.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Exported {len(frames)} RGB/depth frames to {output_dir}")
    print(f"Suggested depth_trunc: {depth_trunc:.4f}")


if __name__ == "__main__":
    main()
