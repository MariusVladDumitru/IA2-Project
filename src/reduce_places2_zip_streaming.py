#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import mimetypes
import os
import random
import sys
import tarfile
import zipfile
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"
}


def is_image_name(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return True

    mime, _ = mimetypes.guess_type(name)
    return mime is not None and mime.startswith("image/")


def bytes_from_gb(gb: float) -> int:
    return int(gb * (1024 ** 3))


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def unique_flat_name(original_name: str, used_names: set[str]) -> str:
    base_name = Path(original_name).name

    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    short_hash = hashlib.sha1(original_name.encode("utf-8")).hexdigest()[:10]
    candidate = f"{stem}_{short_hash}{suffix}"

    counter = 1
    while candidate in used_names:
        candidate = f"{stem}_{short_hash}_{counter}{suffix}"
        counter += 1

    used_names.add(candidate)
    return candidate


def collect_image_members(zip_path: Path) -> list[zipfile.ZipInfo]:
    members = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if is_image_name(info.filename):
                members.append(info)
    return members


def choose_random_subset(
    members: list[zipfile.ZipInfo],
    target_bytes: int,
    seed: int | None,
) -> tuple[list[zipfile.ZipInfo], int]:
    rng = random.Random(seed)
    shuffled = members[:]
    rng.shuffle(shuffled)

    selected = []
    total = 0

    for info in shuffled:
        size = info.file_size

        if not selected and size > target_bytes:
            selected.append(info)
            total += size
            break

        if total + size <= target_bytes:
            selected.append(info)
            total += size

        if total >= target_bytes:
            break

    return selected, total


def write_selected_to_flat_tar(
    zip_path: Path,
    selected: list[zipfile.ZipInfo],
    output_tar: Path,
) -> None:
    used_names: set[str] = set()

    with zipfile.ZipFile(zip_path, "r") as zf, tarfile.open(output_tar, "w") as tf:
        total = len(selected)

        for idx, info in enumerate(selected, start=1):
            flat_name = unique_flat_name(info.filename, used_names)
            arcname = f"places2_random_flat/{flat_name}"

            with zf.open(info, "r") as src:
                data = src.read()

            tar_info = tarfile.TarInfo(name=arcname)
            tar_info.size = len(data)
            tf.addfile(tar_info, io.BytesIO(data))

            if idx % 1000 == 0 or idx == total:
                print(f"[write] {idx}/{total} files written", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly reduce a ZIP image dataset and write directly to a flat TAR archive."
    )
    parser.add_argument("--input-zip", required=True, help="Path to the input ZIP archive")
    parser.add_argument("--output-tar", required=True, help="Path to the output TAR archive")
    parser.add_argument("--target-gb", type=float, default=12.5, help="Target raw image size in GB")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_zip = Path(args.input_zip).expanduser().resolve()
    output_tar = Path(args.output_tar).expanduser().resolve()
    target_bytes = bytes_from_gb(args.target_gb)

    if not input_zip.exists():
        print(f"[error] Input ZIP does not exist: {input_zip}", file=sys.stderr)
        return 1

    output_tar.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Input ZIP: {input_zip}")
    print(f"[info] Output TAR: {output_tar}")
    print(f"[info] Target size: {args.target_gb} GB")
    print(f"[info] Seed: {args.seed}")

    members = collect_image_members(input_zip)
    if not members:
        print("[error] No images found inside ZIP archive.", file=sys.stderr)
        return 1

    print(f"[info] Found {len(members):,} images in ZIP")

    selected, total_size = choose_random_subset(members, target_bytes, args.seed)
    if not selected:
        print("[error] No images selected.", file=sys.stderr)
        return 1

    print(f"[info] Selected {len(selected):,} images")
    print(f"[info] Total selected raw size: {human_size(total_size)}")

    write_selected_to_flat_tar(input_zip, selected, output_tar)

    final_size = output_tar.stat().st_size
    print(f"[done] TAR created: {output_tar}")
    print(f"[done] Final TAR size: {human_size(final_size)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())