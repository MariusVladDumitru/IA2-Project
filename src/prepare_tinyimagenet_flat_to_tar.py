#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATASET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"
ZIP_NAME = "tiny-imagenet-200.zip"
EXTRACTED_DIR_NAME = "tiny-imagenet-200"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jpeg"
}


def is_image_file(path: Path) -> bool:
    if not path.is_file():
        return False

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return True

    mime, _ = mimetypes.guess_type(str(path))
    return mime is not None and mime.startswith("image/")


def md5_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, dst.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None

        downloaded = 0
        chunk_size = 1024 * 1024

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            if total_bytes:
                pct = 100.0 * downloaded / total_bytes
                print(
                    f"\r[download] {human_size(downloaded)} / {human_size(total_bytes)} ({pct:.2f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r[download] {human_size(downloaded)}", end="", flush=True)

    print()


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def collect_train_images(extracted_root: Path) -> list[Path]:
    train_root = extracted_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_root}")

    images: list[Path] = []
    for class_dir in sorted(train_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_images_dir = class_dir / "images"
        if not class_images_dir.is_dir():
            continue
        for img in sorted(class_images_dir.iterdir()):
            if is_image_file(img):
                images.append(img)

    return images


def collect_val_images(extracted_root: Path) -> list[Path]:
    val_images_dir = extracted_root / "val" / "images"
    if not val_images_dir.is_dir():
        raise FileNotFoundError(f"Missing val images directory: {val_images_dir}")

    return [p for p in sorted(val_images_dir.iterdir()) if is_image_file(p)]


def collect_test_images(extracted_root: Path) -> list[Path]:
    test_images_dir = extracted_root / "test" / "images"
    if not test_images_dir.is_dir():
        raise FileNotFoundError(f"Missing test images directory: {test_images_dir}")

    return [p for p in sorted(test_images_dir.iterdir()) if is_image_file(p)]


def write_index_json(index_path: Path, mapping: dict[str, str]) -> None:
    payload = {
        "length": len(mapping),
        "files": mapping,
    }
    index_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def flatten_and_rename(
    images: list[Path],
    output_dir: Path,
    subset_name: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    index_mapping: dict[str, str] = {}
    total_bytes = 0

    for idx, src in enumerate(images):
        suffix = src.suffix
        if not suffix:
            suffix = ".jpg"

        dst_name = f"{idx}{suffix}"
        dst_path = output_dir / dst_name

        shutil.copy2(src, dst_path)

        rel_name = f"{subset_name}/{dst_name}"
        index_mapping[str(idx)] = rel_name
        total_bytes += dst_path.stat().st_size

        if (idx + 1) % 1000 == 0:
            print(
                f"[{subset_name}] copied {idx + 1:,}/{len(images):,} ({human_size(total_bytes)})",
                flush=True,
            )

    print(f"[{subset_name}] done: {len(images):,} images, {human_size(total_bytes)}")
    return index_mapping


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_tar_from_dir(source_dir: Path, output_tar: Path) -> None:
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_tar, "w") as tf:
        for path in sorted(source_dir.rglob("*")):
            arcname = path.relative_to(source_dir)
            tf.add(path, arcname=str(arcname))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Tiny ImageNet, flatten it into train/ and val/, write JSON index files, "
            "and package the processed dataset into a TAR archive."
        )
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Working directory where the ZIP and extracted original dataset will live.",
    )
    parser.add_argument(
        "--output-tar",
        required=True,
        help="Path to the final output TAR archive.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded ZIP after processing.",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep the extracted original dataset after processing.",
    )
    parser.add_argument(
        "--keep-processed-dir",
        action="store_true",
        help="Keep the temporary processed flat directory after creating the TAR.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    output_tar = Path(args.output_tar).expanduser().resolve()

    zip_path = work_dir / ZIP_NAME
    extracted_root = work_dir / EXTRACTED_DIR_NAME

    work_dir.mkdir(parents=True, exist_ok=True)
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    processed_tmp_ctx = tempfile.TemporaryDirectory(prefix="tinyimagenet_flat_")
    processed_root = Path(processed_tmp_ctx.name) / "tinyimagenet_flat"
    processed_train_dir = processed_root / "train"
    processed_val_dir = processed_root / "val"
    train_index_path = processed_root / "train_index.json"
    val_index_path = processed_root / "val_index.json"

    try:
        if not zip_path.exists():
            print(f"[info] Downloading dataset from: {DATASET_URL}")
            download_file(DATASET_URL, zip_path)
        else:
            print(f"[info] ZIP already exists, skipping download: {zip_path}")

        md5 = md5_of_file(zip_path)
        print(f"[info] ZIP MD5: {md5}")
        if md5 != DATASET_MD5:
            print(
                f"[error] MD5 mismatch. Expected {DATASET_MD5}, got {md5}.",
                file=sys.stderr,
            )
            return 1

        if not extracted_root.exists():
            print(f"[info] Extracting ZIP to: {work_dir}")
            extract_zip(zip_path, work_dir)
        else:
            print(f"[info] Extracted dataset already exists, skipping extraction: {extracted_root}")

        train_images = collect_train_images(extracted_root)
        val_images = collect_val_images(extracted_root)
        test_images = collect_test_images(extracted_root)

        if not train_images:
            print("[error] No train images found.", file=sys.stderr)
            return 1
        if not val_images:
            print("[error] No validation images found.", file=sys.stderr)
            return 1
        if not test_images:
            print("[error] No test images found.", file=sys.stderr)
            return 1

        print(f"[info] Original train images: {len(train_images):,}")
        print(f"[info] Original val images:   {len(val_images):,}")
        print(f"[info] Original test images:  {len(test_images):,}")

        combined_val_images = val_images + test_images
        print(f"[info] Combined processed val images (val+test): {len(combined_val_images):,}")

        ensure_empty_dir(processed_root)
        ensure_empty_dir(processed_train_dir)
        ensure_empty_dir(processed_val_dir)

        train_index = flatten_and_rename(
            images=train_images,
            output_dir=processed_train_dir,
            subset_name="train",
        )

        val_index = flatten_and_rename(
            images=combined_val_images,
            output_dir=processed_val_dir,
            subset_name="val",
        )

        write_index_json(train_index_path, train_index)
        write_index_json(val_index_path, val_index)

        print(f"[info] Creating final TAR archive: {output_tar}")
        create_tar_from_dir(processed_root, output_tar)

        print("[done] Final archive created successfully")
        print(f"[done] Output TAR: {output_tar}")
        print(f"[done] Output TAR size: {human_size(output_tar.stat().st_size)}")

        if not args.keep_extracted and extracted_root.exists():
            print(f"[cleanup] Removing extracted original dataset: {extracted_root}")
            shutil.rmtree(extracted_root)

        if not args.keep_zip and zip_path.exists():
            print(f"[cleanup] Removing ZIP: {zip_path}")
            zip_path.unlink()

        if args.keep_processed_dir:
            persistent_dir = output_tar.parent / "tinyimagenet_flat_processed"
            if persistent_dir.exists():
                shutil.rmtree(persistent_dir)
            shutil.copytree(processed_root, persistent_dir)
            print(f"[cleanup] Kept processed directory copy at: {persistent_dir}")

        return 0

    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    finally:
        processed_tmp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())