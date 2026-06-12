from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(r"D:\大三下\CV\hw3")
ROOT_CVHW3 = Path(r"D:\cvhw3")

PATHS = {
    "background": ROOT_CVHW3 / "output" / "background_counter_2dgs_7k" / "train" / "ours_7000" / "fuse_post.ply",
    "object_a": ROOT_CVHW3 / "objectA_server_submission" / "mesh" / "largest_300k.ply",
    "object_b": ROOT_CVHW3 / "task1_assets_ascii" / "objectB_mesh" / "model.obj",
    "object_c": ROOT_CVHW3 / "task1_assets_ascii" / "objectC_mesh" / "model.obj",
}


def mesh_stats(path: Path) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    aabb = mesh.get_axis_aligned_bounding_box()
    verts = np.asarray(mesh.vertices)
    y = verts[:, 1]
    hist, bin_edges = np.histogram(y, bins=80)
    top_bins = np.argsort(hist)[-5:][::-1]
    peaks = []
    for idx in top_bins:
        peaks.append(
            {
                "y_center": float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5),
                "count": int(hist[idx]),
            }
        )
    return {
        "path": str(path),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "aabb_min": aabb.min_bound.tolist(),
        "aabb_max": aabb.max_bound.tolist(),
        "extent": aabb.get_extent().tolist(),
        "center": aabb.get_center().tolist(),
        "y_peaks": peaks,
    }


def main() -> None:
    out_dir = ROOT / "task1_scene_fusion" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {name: mesh_stats(path) for name, path in PATHS.items()}
    out_path = out_dir / "mesh_stats.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
