from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def sharpness(gray_image):
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()


def resize_to_max_side(image, max_side):
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def save_jpg(path: Path, image):
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed to encode frame for {path}")
    path.write_bytes(buffer.tobytes())


def extract_from_video(
    video_path: Path,
    label: str,
    output_dir: Path,
    frames: int,
    max_side: int,
    samples_per_bin: int,
    trim_start: float,
    trim_end: float,
    start_index: int,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0:
        raise RuntimeError(f"Video reports no frames: {video_path}")

    usable_start = max(0, int(frame_count * trim_start))
    usable_end = min(frame_count - 1, int(frame_count * (1.0 - trim_end)))
    span = max(1, usable_end - usable_start)
    frames_to_write = min(frames, span)

    saved = []
    next_index = start_index
    for local_idx in range(frames_to_write):
        bin_start = usable_start + int(local_idx * span / frames_to_write)
        bin_end = usable_start + int((local_idx + 1) * span / frames_to_write)
        if bin_end <= bin_start:
            bin_end = bin_start + 1

        candidates = []
        sample_count = max(1, samples_per_bin)
        for sample_idx in range(sample_count):
            alpha = (sample_idx + 0.5) / sample_count
            frame_idx = min(frame_count - 1, bin_start + int((bin_end - bin_start) * alpha))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            candidates.append((sharpness(gray), frame_idx, frame))

        if not candidates:
            continue

        _, frame_idx, best_frame = max(candidates, key=lambda item: item[0])
        best_frame = resize_to_max_side(best_frame, max_side)
        output_name = f"{next_index:04d}_{label}.jpg"
        output_path = output_dir / output_name
        save_jpg(output_path, best_frame)
        saved.append((output_name, frame_idx))
        next_index += 1

    cap.release()

    if not saved:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    first_frame = cv2.imread(str(output_dir / saved[0][0]))
    height, width = first_frame.shape[:2]
    return {
        "video": str(video_path),
        "label": label,
        "frame_count": frame_count,
        "fps": fps,
        "saved_count": len(saved),
        "width": width,
        "height": height,
        "first_source_frame": saved[0][1],
        "last_source_frame": saved[-1][1],
        "next_index": next_index,
    }


def default_labels(videos):
    if len(videos) == 1:
        return ["obj"]
    labels = []
    for idx, video in enumerate(videos):
        stem = video.stem.lower()
        short = "".join(part[0] for part in stem.replace("-", "_").split("_") if part)
        labels.append(short or f"v{idx + 1}")
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Extract sharp, evenly spaced frames for Object A from one or more videos."
    )
    parser.add_argument("--videos", nargs="+", required=True, help="One or more input videos.")
    parser.add_argument("--dataset", required=True, help="Dataset root, e.g. D:\\cvhw3\\data\\objectA")
    parser.add_argument("--labels", nargs="*", help="Optional labels matching the number of videos.")
    parser.add_argument("--frames-per-video", type=int, default=72)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--samples-per-bin", type=int, default=5)
    parser.add_argument("--trim-start", type=float, default=0.02)
    parser.add_argument("--trim-end", type=float, default=0.02)
    parser.add_argument("--clean", action="store_true", help="Remove existing extracted frames first.")
    args = parser.parse_args()

    videos = [Path(video) for video in args.videos]
    labels = args.labels if args.labels else default_labels(videos)
    if len(labels) != len(videos):
        raise ValueError("--labels must match the number of --videos")

    dataset_path = Path(args.dataset)
    image_dir = dataset_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for old_frame in image_dir.glob("*.jpg"):
            old_frame.unlink()

    summaries = []
    next_index = 0
    for video, label in zip(videos, labels):
        summary = extract_from_video(
            video_path=video,
            label=label,
            output_dir=image_dir,
            frames=args.frames_per_video,
            max_side=args.max_side,
            samples_per_bin=args.samples_per_bin,
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            start_index=next_index,
        )
        next_index = summary.pop("next_index")
        summaries.append(summary)

    print(f"Saved {next_index} total frames to {image_dir}")
    for summary in summaries:
        print(
            f"[{summary['label']}] {summary['video']} -> {summary['saved_count']} frames "
            f"({summary['width']}x{summary['height']}), "
            f"source frames {summary['first_source_frame']}..{summary['last_source_frame']}"
        )


if __name__ == "__main__":
    main()
