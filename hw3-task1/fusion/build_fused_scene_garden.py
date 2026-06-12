from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
GS_REPO = Path(os.environ.get("GS_REPO", r"D:\cvhw3\2d-gaussian-splatting"))
sys.path.append(str(GS_REPO))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary


OUT_DIR = PACKAGE_ROOT / "fusion" / "outputs"
FRAMES_DIR = OUT_DIR / "frames"

BACKGROUND_PATH = PACKAGE_ROOT / "background" / "reconstruction" / "fuse_post.ply"
SPARSE_DIR = PACKAGE_ROOT / "fusion" / "inputs" / "background_sparse" / "0"
OBJECTS = {
    "object_a": PACKAGE_ROOT / "fusion" / "inputs" / "object_meshes" / "objectA_largest_300k.ply",
    "object_b": PACKAGE_ROOT / "fusion" / "inputs" / "object_meshes" / "objectB" / "model.obj",
    "object_c": PACKAGE_ROOT / "fusion" / "inputs" / "object_meshes" / "objectC" / "model.obj",
}

# Estimated from the reconstructed garden mesh using the tabletop band.
TABLE_TOP_Y = 2.08
SCENE_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)

BACKGROUND_CROP_MIN = np.array([-2.1, -0.1, -1.0], dtype=np.float64)
BACKGROUND_CROP_MAX = np.array([3.2, 3.9, 3.5], dtype=np.float64)

# Placement notes:
# - The visible tabletop region is the front band around z ~= 0.35.
# - A is a flatter plush/pillow-like object, so it should stay lower but wider.
# - B should stand upright and remain the smallest object.
# - C is a cup-like object and should be the tallest of the three.
PLACEMENTS = {
    "object_a": {
        "target_height": 0.26,
        "position_xz": [0.10, 0.48],
        "rotation_deg_xyz": [30.0, 0.0, 0.0],
        "align_percentile_low": 10.0,
        "align_percentile_high": 90.0,
        "base_plane_alignment": True,
        "table_clearance": 0.00,
    },
    "object_b": {
        "target_height": 0.44,
        "position_xz": [1.04, 0.44],
        "rotation_deg_xyz": [-58.0, 180.0, 0.0],
        "align_percentile_low": 5.0,
        "align_percentile_high": 95.0,
        "uniform_color": [0.95, 0.72, 0.22],
        "table_clearance": 0.24,
    },
    "object_c": {
        "target_height": 0.56,
        "position_xz": [0.55, 0.14],
        "rotation_deg_xyz": [122.0, 0.0, 0.0],
        "align_percentile_low": 5.0,
        "align_percentile_high": 95.0,
        "table_clearance": 0.08,
    },
}


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def clean_background(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    bbox = o3d.geometry.AxisAlignedBoundingBox(BACKGROUND_CROP_MIN, BACKGROUND_CROP_MAX)
    mesh = mesh.crop(bbox)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    norm_cross = np.linalg.norm(cross)
    if norm_cross < 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - source * np.dot(axis, source)
        axis = axis / np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / (norm_cross**2))


def align_mesh_base_plane(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    points = o3d.geometry.PointCloud()
    points.points = mesh.vertices
    plane_model, inliers = points.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=3000)
    normal = np.asarray(plane_model[:3], dtype=np.float64)
    if np.dot(normal, SCENE_UP) < 0.0:
        normal = -normal
    rotation = rotation_from_vectors(normal, SCENE_UP)
    mesh.rotate(rotation, center=(0.0, 0.0, 0.0))

    vertices = np.asarray(mesh.vertices)
    rotated_inliers = vertices[inliers]
    scene_up_axis = SCENE_UP[1]
    if scene_up_axis < 0.0 and np.percentile(vertices[:, 1], 40.0) > np.percentile(rotated_inliers[:, 1], 50.0):
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi, 0.0)), center=(0.0, 0.0, 0.0))
    if scene_up_axis > 0.0 and np.percentile(vertices[:, 1], 60.0) < np.percentile(rotated_inliers[:, 1], 50.0):
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((0.0, np.pi, 0.0)), center=(0.0, 0.0, 0.0))
    return mesh


def transform_object(mesh: o3d.geometry.TriangleMesh, cfg: dict) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(mesh.vertices)
    mesh.translate(-np.percentile(vertices, 50.0, axis=0))

    if cfg.get("base_plane_alignment"):
        mesh = align_mesh_base_plane(mesh)

    rx, ry, rz = np.deg2rad(cfg["rotation_deg_xyz"])
    rotation = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
    mesh.rotate(rotation, center=(0.0, 0.0, 0.0))

    vertices = np.asarray(mesh.vertices)
    lo = np.percentile(vertices, cfg["align_percentile_low"], axis=0)
    hi = np.percentile(vertices, cfg["align_percentile_high"], axis=0)
    extent = hi - lo
    scale = cfg["target_height"] / max(extent[1], 1e-6)
    mesh.scale(scale, center=(0.0, 0.0, 0.0))

    vertices = np.asarray(mesh.vertices)
    lo = np.percentile(vertices, cfg["align_percentile_low"], axis=0)
    hi = np.percentile(vertices, cfg["align_percentile_high"], axis=0)
    x_target, z_target = cfg["position_xz"]
    if SCENE_UP[1] < 0.0:
        y_shift = TABLE_TOP_Y - hi[1] - cfg.get("table_clearance", 0.002)
    else:
        y_shift = TABLE_TOP_Y - lo[1] + cfg.get("table_clearance", 0.002)
    translation = np.array(
        [
            x_target - (lo[0] + hi[0]) * 0.5,
            y_shift,
            z_target - (lo[2] + hi[2]) * 0.5,
        ],
        dtype=np.float64,
    )
    mesh.translate(translation)
    vertices = np.asarray(mesh.vertices)
    clearance = cfg.get("table_clearance", 0.002)
    if SCENE_UP[1] < 0.0:
        max_y = float(np.max(vertices[:, 1]))
        desired_max_y = TABLE_TOP_Y - clearance
        if max_y > desired_max_y:
            mesh.translate((0.0, desired_max_y - max_y, 0.0))
    else:
        min_y = float(np.min(vertices[:, 1]))
        desired_min_y = TABLE_TOP_Y + clearance
        if min_y < desired_min_y:
            mesh.translate((0.0, desired_min_y - min_y, 0.0))
    if "uniform_color" in cfg:
        mesh.paint_uniform_color(np.asarray(cfg["uniform_color"], dtype=np.float64))
    mesh.compute_vertex_normals()
    return mesh


def save_video(frame_paths: list[Path], out_path: Path, fps: int = 24) -> None:
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for path in frame_paths:
        frame = cv2.imread(str(path))
        writer.write(frame)
    writer.release()


def iter_orbit_views():
    extrinsics = read_extrinsics_binary(str(SPARSE_DIR / "images.bin"))
    intrinsics = read_intrinsics_binary(str(SPARSE_DIR / "cameras.bin"))
    items = sorted(extrinsics.items(), key=lambda kv: kv[1].name)
    for _, img in items:
        cam = intrinsics[img.camera_id]
        width, height = cam.width, cam.height
        if width > 1600:
            scale = 1600.0 / width
            width = int(width * scale)
            height = int(height * scale)
            fx = cam.params[0] * scale
            fy = cam.params[1] * scale
            cx = cam.params[2] * scale
            cy = cam.params[3] * scale
        else:
            fx, fy, cx, cy = cam.params[:4]
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = qvec_to_rotmat(np.asarray(img.qvec, dtype=np.float64))
        extrinsic[:3, 3] = np.asarray(img.tvec, dtype=np.float64)
        yield img.name, width, height, fx, fy, cx, cy, extrinsic


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    background = clean_background(read_mesh(BACKGROUND_PATH))
    transformed = {name: transform_object(read_mesh(path), PLACEMENTS[name]) for name, path in OBJECTS.items()}

    render_meshes = [background] + [transformed[name] for name in OBJECTS]

    fused = o3d.geometry.TriangleMesh(background)
    for mesh in transformed.values():
        fused += mesh
    fused.compute_vertex_normals()

    fused_mesh_path = OUT_DIR / "task1_fused_scene_garden.ply"
    o3d.io.write_triangle_mesh(str(fused_mesh_path), fused, write_vertex_colors=True)

    scene_info = {
        "background": str(BACKGROUND_PATH),
        "table_top_y": TABLE_TOP_Y,
        "placements": PLACEMENTS,
        "fused_mesh": str(fused_mesh_path),
    }
    (OUT_DIR / "scene_config.json").write_text(json.dumps(scene_info, indent=2, ensure_ascii=False), encoding="utf-8")

    frame_paths: list[Path] = []
    first_view = True
    vis = None
    for idx, (name, width, height, fx, fy, cx, cy, extrinsic) in enumerate(iter_orbit_views()):
        if first_view:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="task1_fusion_garden", width=width, height=height, visible=False)
            for mesh in render_meshes:
                vis.add_geometry(mesh)
            render_option = vis.get_render_option()
            render_option.background_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            render_option.mesh_show_back_face = True
            render_option.light_on = True
            render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
            first_view = False

        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        params.extrinsic = extrinsic
        vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        image = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255.0).clip(0, 255).astype(np.uint8)
        frame_path = FRAMES_DIR / f"{idx:04d}_{Path(name).stem}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)

    save_video(frame_paths, OUT_DIR / "task1_fused_scene_garden.mp4", fps=24)
    if vis is not None:
        vis.destroy_window()

    print(f"Saved fused mesh to: {fused_mesh_path}")
    print(f"Saved video to: {OUT_DIR / 'task1_fused_scene_garden.mp4'}")


if __name__ == "__main__":
    main()
