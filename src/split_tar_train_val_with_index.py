#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import random
import sys
import tarfile
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


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} TB"


def add_bytes_to_tar(
    dst_tf: tarfile.TarFile,
    arcname: str,
    data: bytes,
    mode: int = 0o644,
) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = mode
    dst_tf.addfile(info, io.BytesIO(data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read images from an input TAR archive, randomly split them into "
            "train/val subsets, rename them from 0 in each subset, and write "
            "a new TAR archive with train_index.json and val_index.json."
        )
    )

    parser.add_argument(
        "--input-tar",
        required=True,
        help="Path to the input TAR archive, e.g. places2_random_flat.tar",
    )
    parser.add_argument(
        "--output-tar",
        required=True,
        help="Path to the output TAR archive",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio in (0,1). Default: 0.8",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_tar = Path(args.input_tar).expanduser().resolve()
    output_tar = Path(args.output_tar).expanduser().resolve()
    train_ratio = args.train_ratio
    seed = args.seed

    if not input_tar.is_file():
        print(f"[error] Input TAR does not exist: {input_tar}", file=sys.stderr)
        return 1

    if output_tar == input_tar:
        print("[error] Output TAR must be different from input TAR.", file=sys.stderr)
        return 1

    if not (0.0 < train_ratio < 1.0):
        print("[error] --train-ratio must be between 0 and 1.", file=sys.stderr)
        return 1

    output_tar.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(input_tar, "r") as src_tf:
            members = [
                m for m in src_tf.getmembers()
                if m.isfile() and is_image_name(m.name)
            ]

        if not members:
            print("[error] No image files found inside input TAR.", file=sys.stderr)
            return 1

        total_images = len(members)
        print(f"[info] Found {total_images:,} images in input TAR")

        rng = random.Random(seed)
        rng.shuffle(members)

        train_count = int(total_images * train_ratio)
        val_count = total_images - train_count

        train_members = members[:train_count]
        val_members = members[train_count:]

        print(f"[info] Random seed: {seed}")
        print(f"[info] Planned train images: {len(train_members):,}")
        print(f"[info] Planned val images:   {len(val_members):,}")

        written_train = 0
        written_val = 0
        bytes_train = 0
        bytes_val = 0

        train_index_files: dict[str, str] = {}
        val_index_files: dict[str, str] = {}

        with tarfile.open(input_tar, "r") as src_tf, tarfile.open(output_tar, "w") as dst_tf:
            for member in train_members:
                suffix = Path(member.name).suffix.lower()
                if not suffix:
                    suffix = ".jpg"

                new_name = f"train/{written_train}{suffix}"

                extracted = src_tf.extractfile(member)
                if extracted is None:
                    print(f"[warn] Skipping unreadable train member: {member.name}")
                    continue

                data = extracted.read()

                info = tarfile.TarInfo(name=new_name)
                info.size = len(data)
                info.mtime = member.mtime
                info.mode = 0o644
                dst_tf.addfile(info, io.BytesIO(data))

                train_index_files[str(written_train)] = new_name
                written_train += 1
                bytes_train += len(data)

                if written_train % 1000 == 0:
                    print(
                        f"[train] {written_train:,}/{len(train_members):,} written "
                        f"({human_size(bytes_train)})",
                        flush=True,
                    )

            for member in val_members:
                suffix = Path(member.name).suffix.lower()
                if not suffix:
                    suffix = ".jpg"

                new_name = f"val/{written_val}{suffix}"

                extracted = src_tf.extractfile(member)
                if extracted is None:
                    print(f"[warn] Skipping unreadable val member: {member.name}")
                    continue

                data = extracted.read()

                info = tarfile.TarInfo(name=new_name)
                info.size = len(data)
                info.mtime = member.mtime
                info.mode = 0o644
                dst_tf.addfile(info, io.BytesIO(data))

                val_index_files[str(written_val)] = new_name
                written_val += 1
                bytes_val += len(data)

                if written_val % 1000 == 0:
                    print(
                        f"[val] {written_val:,}/{len(val_members):,} written "
                        f"({human_size(bytes_val)})",
                        flush=True,
                    )

            train_index_payload = {
                "length": written_train,
                "files": train_index_files,
            }
            val_index_payload = {
                "length": written_val,
                "files": val_index_files,
            }

            train_index_bytes = json.dumps(
                train_index_payload,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")

            val_index_bytes = json.dumps(
                val_index_payload,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")

            add_bytes_to_tar(dst_tf, "train_index.json", train_index_bytes)
            add_bytes_to_tar(dst_tf, "val_index.json", val_index_bytes)

        final_size = output_tar.stat().st_size

        print("[done] Finished creating split archive with JSON index files")
        print(f"[done] Output TAR: {output_tar}")
        print(f"[done] Output TAR size: {human_size(final_size)}")
        print(f"[done] Final train count: {written_train:,}")
        print(f"[done] Final val count:   {written_val:,}")
        print(f"[done] Train bytes: {human_size(bytes_train)}")
        print(f"[done] Val bytes:   {human_size(bytes_val)}")
        print("[done] Added files: train_index.json, val_index.json")

        return 0

    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())