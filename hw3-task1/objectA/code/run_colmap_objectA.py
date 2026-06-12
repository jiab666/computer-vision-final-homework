from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from typing import Dict, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def copy_images(source_dir: Path, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    image_names = []
    for image_path in sorted(source_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        target_path = target_dir / image_path.name
        if not target_path.exists() or image_path.stat().st_mtime > target_path.stat().st_mtime:
            shutil.copy2(image_path, target_path)
        image_names.append(image_path.name)
    if not image_names:
        raise RuntimeError(f"No images found in {source_dir}")
    return image_names


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    import struct

    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file: str) -> Dict[int, object]:
    import collections

    Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = binary_image_properties[1:5]
            tvec = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, 1)
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=None,
                point3D_ids=None,
            )
    return images


def read_points3d_binary(path_to_model_file: str) -> Dict[int, object]:
    import collections

    Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error"])
    points3d = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            point_line = read_next_bytes(fid, 43, "QdddBBBd")
            point3d_id = point_line[0]
            xyz = point_line[1:4]
            rgb = point_line[4:7]
            error = point_line[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(track_length * 8, 1)
            points3d[point3d_id] = Point3D(id=point3d_id, xyz=xyz, rgb=rgb, error=error)
    return points3d


def model_score(model_dir: Path) -> Tuple[int, int]:
    images_path = model_dir / "images.bin"
    points_path = model_dir / "points3D.bin"
    if not images_path.exists() or not points_path.exists():
        return (0, 0)
    return (len(read_images_binary(str(images_path))), len(read_points3d_binary(str(points_path))))


def choose_largest_model(sparse_parent: Path) -> Path:
    candidates = [path for path in sparse_parent.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError("COLMAP did not reconstruct any sparse model.")
    scored = [(model_score(path), path) for path in candidates]
    score, best = max(scored, key=lambda item: item[0])
    if score == (0, 0):
        raise RuntimeError("All sparse models are empty.")
    print(f"Selected sparse model {best.name}: {score[0]} registered images, {score[1]} 3D points")
    return best


def ensure_sparse_zero(dataset: Path) -> Path:
    sparse_root = dataset / "sparse"
    sparse_zero = sparse_root / "0"
    if sparse_zero.exists():
        return sparse_zero

    cameras_bin = sparse_root / "cameras.bin"
    images_bin = sparse_root / "images.bin"
    points3d_bin = sparse_root / "points3D.bin"
    if cameras_bin.exists() and images_bin.exists() and points3d_bin.exists():
        sparse_zero.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cameras_bin), str(sparse_zero / "cameras.bin"))
        shutil.move(str(images_bin), str(sparse_zero / "images.bin"))
        shutil.move(str(points3d_bin), str(sparse_zero / "points3D.bin"))
        return sparse_zero

    raise RuntimeError(f"Expected sparse reconstruction under {sparse_root}, but it was not created.")


def main():
    parser = argparse.ArgumentParser(description="Run COLMAP with stronger Object A settings and pick the best model.")
    parser.add_argument("--dataset", required=True, help="Dataset root, e.g. D:\\cvhw3\\data\\objectA")
    parser.add_argument("--source-images", default=None, help="Defaults to <dataset>\\images")
    parser.add_argument("--colmap-executable", required=True)
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--max-num-features", type=int, default=12000)
    parser.add_argument("--peak-threshold", type=float, default=0.004)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    source_images = Path(args.source_images) if args.source_images else dataset / "images"
    input_dir = dataset / "input"
    distorted_dir = dataset / "distorted"
    database_path = distorted_dir / "database.db"
    sparse_parent = distorted_dir / "sparse"
    best_sparse = distorted_dir / "sparse_best"

    dataset.mkdir(parents=True, exist_ok=True)
    image_names = copy_images(source_images, input_dir)

    if args.reset:
        reset_path(database_path)
        reset_path(sparse_parent)
        reset_path(best_sparse)
        reset_path(dataset / "images")
        reset_path(dataset / "sparse")

    distorted_dir.mkdir(parents=True, exist_ok=True)
    sparse_parent.mkdir(parents=True, exist_ok=True)

    use_gpu = "1" if args.use_gpu else "0"
    colmap = args.colmap_executable

    run_command(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(input_dir),
            "--ImageReader.camera_model",
            args.camera_model,
            "--ImageReader.single_camera",
            "1",
            "--FeatureExtraction.use_gpu",
            use_gpu,
            "--FeatureExtraction.max_image_size",
            "1600",
            "--SiftExtraction.max_num_features",
            str(args.max_num_features),
            "--SiftExtraction.peak_threshold",
            str(args.peak_threshold),
        ]
    )

    run_command(
        [
            colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            use_gpu,
            "--FeatureMatching.guided_matching",
            "1",
        ]
    )

    run_command(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(input_dir),
            "--output_path",
            str(sparse_parent),
            "--Mapper.multiple_models",
            "1",
            "--Mapper.min_model_size",
            str(max(8, min(20, len(image_names) // 4))),
            "--Mapper.ba_global_function_tolerance",
            "1e-6",
            "--Mapper.abs_pose_min_num_inliers",
            "20",
            "--Mapper.init_min_num_inliers",
            "80",
            "--Mapper.ba_local_num_images",
            "8",
        ]
    )

    chosen_model = choose_largest_model(sparse_parent)
    reset_path(best_sparse)
    shutil.copytree(chosen_model, best_sparse)

    run_command(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            str(input_dir),
            "--input_path",
            str(best_sparse),
            "--output_path",
            str(dataset),
            "--output_type",
            "COLMAP",
        ]
    )

    final_sparse_zero = ensure_sparse_zero(dataset)
    print(f"Done. Final COLMAP dataset: {dataset}")
    print(f"Final sparse model: {final_sparse_zero}")


if __name__ == "__main__":
    main()
