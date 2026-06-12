from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw


def remove_small_components(mesh: o3d.geometry.TriangleMesh, keep_components: int, min_triangles: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)
    if len(mesh.triangles) == 0:
        return mesh

    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    counts = np.asarray(counts)
    order = np.argsort(counts)[::-1]
    keep = set(order[: max(1, keep_components)].tolist())
    remove = np.array([(cluster not in keep) or (counts[cluster] < min_triangles) for cluster in clusters], dtype=bool)
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    return mesh


def simplify(mesh: o3d.geometry.TriangleMesh, target_triangles: int) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) <= target_triangles:
        return o3d.geometry.TriangleMesh(mesh)
    reduced = mesh.simplify_quadric_decimation(target_triangles)
    reduced.remove_degenerate_triangles()
    reduced.remove_duplicated_triangles()
    reduced.remove_duplicated_vertices()
    reduced.remove_unreferenced_vertices()
    reduced.compute_vertex_normals()
    return reduced


def write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)
    bbox = mesh.get_axis_aligned_bounding_box()
    return {
        "path": path.as_posix(),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "bbox_min": bbox.min_bound.tolist(),
        "bbox_max": bbox.max_bound.tolist(),
    }


def project_points(points: np.ndarray, colors: np.ndarray, view: str, size: int) -> Image.Image:
    axes = {
        "front": ((0, 1), 2),
        "back": ((0, 1), 2),
        "left": ((2, 1), 0),
        "right": ((2, 1), 0),
        "top": ((0, 2), 1),
        "iso": (None, None),
    }
    pts = points.copy()
    if view == "back":
        pts[:, 0] *= -1
    elif view == "right":
        pts[:, 2] *= -1
    elif view == "top":
        pts[:, 1] *= -1
    elif view == "iso":
        yaw = np.deg2rad(38)
        pitch = np.deg2rad(-22)
        ry = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ]
        )
        rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        pts = pts @ (ry @ rx).T

    if view == "iso":
        xy = pts[:, [0, 1]]
        depth = pts[:, 2]
    else:
        xy_axes, depth_axis = axes[view]
        xy = pts[:, list(xy_axes)]
        depth = pts[:, depth_axis]

    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = (size - 24) / span.max()
    centered = (xy - (mins + maxs) / 2.0) * scale
    px = np.round(centered[:, 0] + size / 2).astype(np.int32)
    py = np.round(-centered[:, 1] + size / 2).astype(np.int32)

    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    px, py, depth, colors = px[valid], py[valid], depth[valid], colors[valid]
    order = np.argsort(depth)
    px, py, colors = px[order], py[order], colors[order]

    image = Image.new("RGB", (size, size), "white")
    pixels = image.load()
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    for x, y, color in zip(px, py, rgb):
        c = tuple(int(v) for v in color)
        pixels[x, y] = c
        if x + 1 < size:
            pixels[x + 1, y] = c
        if y + 1 < size:
            pixels[x, y + 1] = c
    return image


def make_preview(mesh: o3d.geometry.TriangleMesh, path: Path, sample_points: int = 180_000) -> None:
    pcd = mesh.sample_points_uniformly(number_of_points=min(sample_points, max(1, len(mesh.triangles))))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.size == 0:
        colors = np.tile(np.array([[0.75, 0.75, 0.75]]), (points.shape[0], 1))

    labels = ["front", "back", "left", "right", "top", "iso"]
    views = [project_points(points, colors, label, 360) for label in labels]
    gap = 14
    label_h = 24
    sheet = Image.new("RGB", (3 * 360 + 2 * gap, 2 * (360 + label_h) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(zip(labels, views)):
        x = (idx % 3) * (360 + gap)
        y = (idx // 3) * (360 + label_h + gap)
        draw.text((x, y + 4), label, fill=(20, 20, 20))
        sheet.paste(image, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cleaner mesh variants and previews for Object A.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-triangles", default=100, type=int)
    args = parser.parse_args()

    output = Path(args.output)
    mesh = o3d.io.read_triangle_mesh(args.input)
    mesh.compute_vertex_normals()

    variants = {}
    configs = {
        "largest": remove_small_components(mesh, 1, args.min_triangles),
        "top3": remove_small_components(mesh, 3, args.min_triangles),
        "top6": remove_small_components(mesh, 6, args.min_triangles),
    }
    configs["clean_top12"] = mesh

    for name, variant in configs.items():
        variants[name] = write_mesh(output / f"{name}.ply", variant)
        make_preview(variant, output / f"{name}_preview.jpg")
        for target in (100_000, 300_000):
            simplified = simplify(variant, target)
            key = f"{name}_{target // 1000}k"
            variants[key] = write_mesh(output / f"{key}.ply", simplified)

    (output / "mesh_variants_summary.json").write_text(json.dumps(variants, indent=2), encoding="utf-8")
    print(f"Wrote mesh variants to {output}")
    print(json.dumps({k: {"vertices": v["vertices"], "triangles": v["triangles"]} for k, v in variants.items()}, indent=2))


if __name__ == "__main__":
    main()
