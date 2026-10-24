"""
UR Fall Detection Dataset (URFD) Preparation Workflow for Google Colab and Local Workstations.

Automates:
1. Downloading RGB image sequence zip archives directly from the official University of Rzeszow repository.
2. Extracting into standardized, naturally-sorted sequence directories:
     data/
       ├── fall/
       │   ├── fall-01-cam0-rgb/
       │   ├── fall-01-cam1-rgb/
       │   └── ...
       └── adl/
           ├── adl-01-cam0-rgb/
           └── ...
3. Frame integrity verification and natural sorting sanity check.
"""

import os
import sys
import re
import zipfile
import argparse
import urllib.request
import urllib.error
from typing import List, Dict, Tuple, Optional, Any
import cv2


URFD_BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data"


def natural_sort_key(s: str) -> list:
    """Natural alphanumeric sort key (e.g. frame_2.png before frame_10.png)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare URFD RGB Sequence Dataset for SafeCCTV Model")
    parser.add_argument("--output-dir", type=str, default="research_model/data",
                        help="Target directory to create 'fall/' and 'adl/' sequence folders")
    parser.add_argument("--zip-dir", type=str, default="research_model/downloads",
                        help="Directory to store downloaded .zip archives")
    parser.add_argument("--max-falls", type=int, default=30,
                        help="Number of fall sequences to process (1 to 30, default: 30)")
    parser.add_argument("--max-adls", type=int, default=40,
                        help="Number of ADL sequences to process (1 to 40, default: 40)")
    parser.add_argument("--include-cam1", action="store_true", default=True,
                        help="Include Camera 1 (ceiling camera) for fall sequences (default: True)")
    parser.add_argument("--no-cam1", dest="include_cam1", action="store_false",
                        help="Exclude Camera 1, only use Camera 0")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip downloading and extract zips already present in --zip-dir")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify an existing extracted dataset directory in --output-dir")
    return parser.parse_args()


def download_file(url: str, dest_path: str) -> bool:
    """Downloads a file with HTTP progress tracking."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  [Cached] {os.path.basename(dest_path)} already exists.")
        return True

    print(f"  [Downloading] {url} -> {os.path.basename(dest_path)} ...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 SafeCCTV Research Client"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 512  # 512 KB

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r    {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.1f}%)")
                    sys.stdout.flush()
            sys.stdout.write("\n")
        return True
    except Exception as e:
        print(f"\n    [Download Failed] {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip_sequence(zip_path: str, target_dir: str) -> int:
    """
    Extracts a URFD sequence zip into target_dir.
    Flattens any internal nesting so that *.png frames sit directly in target_dir.
    Returns number of extracted frame images.
    """
    os.makedirs(target_dir, exist_ok=True)
    extracted_frames = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                fname = os.path.basename(member.filename)
                ext = os.path.splitext(fname)[1].lower()
                if ext in {".png", ".jpg", ".jpeg", ".bmp"}:
                    target_file_path = os.path.join(target_dir, fname)
                    with z.open(member) as source, open(target_file_path, "wb") as target:
                        target.write(source.read())
                    extracted_frames += 1
    except zipfile.BadZipFile as e:
        print(f"  [Error] Corrupted or incomplete zip archive {zip_path}: {e}")
        return 0

    return extracted_frames


def verify_dataset(dataset_dir: str) -> Dict[str, Any]:
    """
    Verifies that the extracted dataset has valid RGB frames and proper structure.
    Checks frame integrity, resolution, and natural chronological sorting.
    """
    print("\n" + "=" * 65)
    print(f"  Verifying URFD Dataset Structure in: {dataset_dir}  ")
    print("=" * 65)

    stats = {
        "fall_sequences": 0,
        "adl_sequences": 0,
        "total_frames": 0,
        "corrupted_images": 0,
        "sequences": []
    }

    for category in ["fall", "adl"]:
        cat_dir = os.path.join(dataset_dir, category)
        if not os.path.exists(cat_dir):
            print(f"  [Notice] Directory not found: {cat_dir}")
            continue

        seq_folders = [
            f for f in os.listdir(cat_dir)
            if os.path.isdir(os.path.join(cat_dir, f))
        ]
        seq_folders.sort(key=natural_sort_key)

        print(f"\n[{category.upper()}] Found {len(seq_folders)} sequence directories:")

        for seq_name in seq_folders:
            seq_path = os.path.join(cat_dir, seq_name)
            frame_files = [
                f for f in os.listdir(seq_path)
                if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}
            ]

            if not frame_files:
                print(f"  [Empty] {seq_name} has 0 images!")
                continue

            # Check natural sorting
            frame_files.sort(key=natural_sort_key)
            first_frame = frame_files[0]
            last_frame = frame_files[-1]

            # Sample first frame to check resolution
            sample_img = cv2.imread(os.path.join(seq_path, first_frame))
            if sample_img is None:
                stats["corrupted_images"] += 1
                res = "CORRUPTED"
            else:
                h, w, c = sample_img.shape
                res = f"{w}x{h} ({c}ch)"

            if category == "fall":
                stats["fall_sequences"] += 1
            else:
                stats["adl_sequences"] += 1
            stats["total_frames"] += len(frame_files)

            stats["sequences"].append({
                "category": category,
                "name": seq_name,
                "frames": len(frame_files),
                "first": first_frame,
                "last": last_frame,
                "resolution": res
            })

            print(f"  • {seq_name:<24} | Frames: {len(frame_files):>4} | {res:<12} | First: {first_frame} | Last: {last_frame}")

    print("\n" + "-" * 65)
    print(f"Summary:")
    print(f"  • Fall sequences:      {stats['fall_sequences']}")
    print(f"  • ADL sequences:       {stats['adl_sequences']}")
    print(f"  • Total frame images:  {stats['total_frames']}")
    print(f"  • Corrupted images:    {stats['corrupted_images']}")
    print("=" * 65)

    return stats


def prepare_dataset(
    output_dir: str,
    zip_dir: str,
    max_falls: int = 30,
    max_adls: int = 40,
    include_cam1: bool = True,
    extract_only: bool = False
):
    """
    Downloads and extracts URFD dataset into:
      output_dir/fall/<sequence_name>/*.png
      output_dir/adl/<sequence_name>/*.png
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(zip_dir, exist_ok=True)

    fall_dir = os.path.join(output_dir, "fall")
    adl_dir = os.path.join(output_dir, "adl")
    os.makedirs(fall_dir, exist_ok=True)
    os.makedirs(adl_dir, exist_ok=True)

    print("=" * 65)
    print("  SafeCCTV Research: URFD RGB Dataset Preparation  ")
    print("=" * 65)
    print(f"Target Output Directory:   {os.path.abspath(output_dir)}")
    print(f"Archive Cache Directory:   {os.path.abspath(zip_dir)}")
    print(f"Sequences Requested:       {max_falls} Falls, {max_adls} ADL")
    print(f"Include Ceiling Camera 1:  {include_cam1}")
    print("=" * 65)

    # 1. Fall sequences
    print("\n--- [Step 1/2] Processing Fall Sequences ---")
    for i in range(1, max_falls + 1):
        # Camera 0 (Frontal)
        seq_cam0 = f"fall-{i:02d}-cam0-rgb"
        zip_name = f"{seq_cam0}.zip"
        zip_path = os.path.join(zip_dir, zip_name)
        target_seq_dir = os.path.join(fall_dir, seq_cam0)

        if not extract_only:
            url = f"{URFD_BASE_URL}/{zip_name}"
            download_file(url, zip_path)

        if os.path.exists(zip_path):
            n_frames = extract_zip_sequence(zip_path, target_seq_dir)
            print(f"  [Extracted] {seq_cam0}: {n_frames} frames -> {target_seq_dir}")

        # Camera 1 (Ceiling)
        if include_cam1:
            seq_cam1 = f"fall-{i:02d}-cam1-rgb"
            zip_name_cam1 = f"{seq_cam1}.zip"
            zip_path_cam1 = os.path.join(zip_dir, zip_name_cam1)
            target_seq_dir_cam1 = os.path.join(fall_dir, seq_cam1)

            if not extract_only:
                url_cam1 = f"{URFD_BASE_URL}/{zip_name_cam1}"
                download_file(url_cam1, zip_path_cam1)

            if os.path.exists(zip_path_cam1):
                n_frames_cam1 = extract_zip_sequence(zip_path_cam1, target_seq_dir_cam1)
                print(f"  [Extracted] {seq_cam1}: {n_frames_cam1} frames -> {target_seq_dir_cam1}")

    # 2. ADL sequences
    print("\n--- [Step 2/2] Processing ADL Sequences ---")
    for i in range(1, max_adls + 1):
        seq_adl = f"adl-{i:02d}-cam0-rgb"
        zip_name = f"{seq_adl}.zip"
        zip_path = os.path.join(zip_dir, zip_name)
        target_seq_dir = os.path.join(adl_dir, seq_adl)

        if not extract_only:
            url = f"{URFD_BASE_URL}/{zip_name}"
            download_file(url, zip_path)

        if os.path.exists(zip_path):
            n_frames = extract_zip_sequence(zip_path, target_seq_dir)
            print(f"  [Extracted] {seq_adl}: {n_frames} frames -> {target_seq_dir}")

    # 3. Verification
    verify_dataset(output_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.verify_only:
        verify_dataset(args.output_dir)
    else:
        prepare_dataset(
            output_dir=args.output_dir,
            zip_dir=args.zip_dir,
            max_falls=args.max_falls,
            max_adls=args.max_adls,
            include_cam1=args.include_cam1,
            extract_only=args.extract_only
        )
