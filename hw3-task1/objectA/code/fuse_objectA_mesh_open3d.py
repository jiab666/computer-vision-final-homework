from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_mesh(mesh: o3d.geometry.TriangleMesh, keep_clusters: int, min_triangles: int) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if cluster_n_triangles.size == 0:
        return mesh

    cluster_order = np.argsort(cluster_n_triangles)[::-1]
    keep = set(cluster_order[: max(1, keep_clusters)].tolist())
    remove = np.array(
        [(cluster not in keep) or (cluster_n_triangles[cluster] < min_triangles) for cluster in triangle_clusters],
        dtype=bool,
    )
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse 2DGS RGB/depth frames into an Open3D mesh.")
    parser.add_argument("--input", required=True, help="Path to transforms_tsdf.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel-size", default=0.0, type=float)
    parser.add_argument("--sdf-trunc", default=0.0, type=float)
    parser.add_argument("--depth-trunc", default=0.0, type=float)
    parser.add_argument("--keep-clusters", default=12, type=int)
    parser.add_argument("--min-triangles", default=100, type=int)
    parser.add_argument("--max-frames", default=0, type=int)
    args = parser.parse_args()

    metadata_path = Path(args.input)
    input_dir = metadata_path.parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path)
    frames = metadata["frames"]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    depth_trunc = args.depth_trunc if args.depth_trunc > 0 else float(metadata["depth_trunc_suggestion"])
    voxel_size = args.voxel_size if args.voxel_size > 0 else depth_trunc / 512.0
    sdf_trunc = args.sdf_trunc if args.sdf_trunc > 0 else voxel_size * 5.0

    print(f"Frames: {len(frames)}")
    print(f"depth_trunc={depth_trunc:.6f}, voxel_size={voxel_size:.6f}, sdf_trunc={sdf_trunc:.6f}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame in tqdm(frames, desc="TSDF integration"):
        color = np.asarray(Image.open(input_dir / frame["color"]).convert("RGB"), dtype=np.uint8)
        depth = np.load(input_dir / frame["depth"]).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0.0] = 0.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(frame["width"]),
            int(frame["height"]),
            float(frame["fx"]),
            float(frame["fy"]),
            float(frame["cx"]),
            float(frame["cy"]),
        )
        extrinsic = np.asarray(frame["extrinsic"], dtype=np.float64)
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    raw_path = output_dir / "objectA_mesh_raw.ply"
    o3d.io.write_triangle_mesh(str(raw_path), mesh)

    cleaned = clean_mesh(mesh, args.keep_clusters, args.min_triangles)
    cleaned.compute_vertex_normals()
    clean_path = output_dir / "objectA_mesh_clean.ply"
    o3d.io.write_triangle_mesh(str(clean_path), cleaned)

    summary = {
        "frames": len(frames),
        "depth_trunc": depth_trunc,
        "voxel_size": voxel_size,
        "sdf_trunc": sdf_trunc,
        "raw_vertices": len(mesh.vertices),
        "raw_triangles": len(mesh.triangles),
        "clean_vertices": len(cleaned.vertices),
        "clean_triangles": len(cleaned.triangles),
        "raw_mesh": raw_path.as_posix(),
        "clean_mesh": clean_path.as_posix(),
    }
    (output_dir / "mesh_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Raw mesh: {raw_path}")
    print(f"Clean mesh: {clean_path}")
    print(f"Clean vertices={summary['clean_vertices']}, triangles={summary['clean_triangles']}")


if __name__ == "__main__":
    main()
