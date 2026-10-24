"""
Dataset loader and sequence extractor for MobileNetV2-LSTM Fall Detection.
Supports both video files (.mp4, .avi, etc.) and extracted RGB image sequences (URFD).
Enforces:
1. Natural sorting of image frames to preserve temporal chronological order.
2. Target FPS downsampling (e.g. 30 FPS source downsampled to 3 FPS).
3. 15-frame temporal sliding windows (with last-frame padding for shorter clips).
4. Strict video-level / subject-level data splitting to guarantee zero data leakage.
"""

import os
import re
import cv2
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import tensorflow as tf
from .config import ResearchModelConfig


def natural_sort_key(s: str) -> List[Any]:
    """
    Key function for natural alphanumeric sorting (e.g., 'frame_2.png' before 'frame_10.png').
    Splits string into numeric and non-numeric chunks so numbers are compared numerically.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


class VideoLevelSplitter:
    """
    Partitions video files and image sequence folders into train, validation,
    and test subsets at the VIDEO/SUBJECT level.
    This prevents temporal autocorrelation, multi-camera viewpoint leakage,
    and subject identity leakage: all sequences from a specific video/subject
    remain strictly within one partition.
    """
    def __init__(
        self,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ):
        assert val_split + test_split < 1.0, "Sum of val_split and test_split must be < 1.0"
        self.val_split = max(0.0, val_split)
        self.test_split = max(0.0, test_split)
        self.random_seed = random_seed

    def split_videos(
        self,
        video_metadata: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits a list of video records [{path: ..., label: 0/1, subject_id: ...}]
        stratified by class and grouped by subject/video entity.
        """
        rng = random.Random(self.random_seed)

        # Group by label to ensure class balance across splits
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for item in video_metadata:
            lbl = int(item["label"])
            by_class.setdefault(lbl, []).append(item)

        train_set: List[Dict[str, Any]] = []
        val_set: List[Dict[str, Any]] = []
        test_set: List[Dict[str, Any]] = []

        for lbl, items in by_class.items():
            # Group by subject / sequence base identifier
            subj_groups: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                sid = it.get("subject_id")
                if not sid:
                    sid = os.path.basename(it["path"].rstrip("/\\"))
                subj_groups.setdefault(sid, []).append(it)

            subjects = list(subj_groups.keys())
            rng.shuffle(subjects)
            n_subj = len(subjects)

            # Compute split counts
            n_test = int(round(n_subj * self.test_split)) if self.test_split > 0 else 0
            if self.test_split > 0 and n_test == 0 and n_subj >= 3:
                n_test = 1

            n_val = int(round(n_subj * self.val_split)) if self.val_split > 0 else 0
            if self.val_split > 0 and n_val == 0 and (n_subj - n_test) >= 2:
                n_val = 1

            # Ensure training split has at least 1 item if n_subj > 0
            if n_subj > 0 and (n_test + n_val) >= n_subj:
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1

            test_subjs = set(subjects[:n_test])
            val_subjs = set(subjects[n_test:n_test + n_val])
            train_subjs = set(subjects[n_test + n_val:])

            for sid, v_list in subj_groups.items():
                if sid in test_subjs:
                    test_set.extend(v_list)
                elif sid in val_subjs:
                    val_set.extend(v_list)
                else:
                    train_set.extend(v_list)

        # Shuffle each final split
        rng.shuffle(train_set)
        rng.shuffle(val_set)
        rng.shuffle(test_set)

        return train_set, val_set, test_set


class VideoSequenceDataset:
    """
    Video sequence dataset extractor.
    Loads video files or extracted image sequence folders (e.g. URFD RGB frames),
    naturally sorts frame files, downsamples to target FPS (e.g. 3 FPS),
    resizes to 224x224 RGB, and slices them into 15-frame temporal sequences.
    """
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, config: Optional[ResearchModelConfig] = None):
        self.config = config if config is not None else ResearchModelConfig()

    @classmethod
    def extract_subject_id(cls, name_or_path: str) -> str:
        """
        Extracts subject or sequence base identifier to prevent leakage.
        For URFD:
            'fall-01-cam0-rgb' and 'fall-01-cam1-rgb' -> 'fall-01'
            'adl-05-cam0-rgb' -> 'adl-05'
        For general datasets:
            'subject02_fall_01' -> 'subject02'
        Also inspects parent directory if name_or_path is nested (e.g. 'subject03/video_1.mp4').
        """
        norm = name_or_path.replace("\\", "/").rstrip("/")
        base = os.path.basename(norm)
        parent = os.path.basename(os.path.dirname(norm))

        # 1. Match URFD sequence tags on basename first (e.g., fall-01, adl-03)
        urfd_match = re.search(r"((?:fall|adl)[\-_]?\d+)", base, re.IGNORECASE)
        if urfd_match:
            return urfd_match.group(1).lower()

        # 2. Match subject tags on basename or parent (e.g., subject-02, person_1, sub3)
        for candidate in [base, parent]:
            subj_match = re.search(r"(?:subject|sub|person|actor)[\-_]?(\d+)", candidate, re.IGNORECASE)
            if subj_match:
                return subj_match.group(0).lower()

        return base.lower()

    @classmethod
    def determine_label(cls, path: str, data_dir: Optional[str] = None) -> int:
        """
        Determines binary label: 1 = Fall, 0 = ADL / Normal.
        Safely inspects relative path within data_dir and target folder/file basename
        without being fooled by ancestor directory names (e.g. 'SafeCCTV_fall_project/data/adl').
        """
        if data_dir:
            try:
                rel_path = os.path.relpath(path, data_dir).replace("\\", "/").lower()
            except ValueError:
                rel_path = os.path.basename(path).lower()
        else:
            rel_path = os.path.basename(path).lower()

        base_name = os.path.basename(path.rstrip("/\\")).lower()
        parts = [p.lower() for p in rel_path.split("/") if p and p != "."]

        # Explicit ADL indicators in relative path or basename take precedence
        adl_tokens = {"adl", "adls", "normal", "no_fall", "negative", "label_0", "non_fall", "not_fall"}
        if any(p in adl_tokens for p in parts) or any(p.startswith("adl") for p in parts) or base_name.startswith("adl"):
            return 0

        # Explicit Fall indicators in relative path or basename
        fall_tokens = {"fall", "falls", "positive", "label_1"}
        if any(p in fall_tokens for p in parts) or any(p.startswith("fall") for p in parts) or base_name.startswith("fall"):
            return 1

        # Fallback keywords
        if any(k in base_name for k in ["adl", "normal", "walk", "sit", "bend", "lie", "stand"]):
            return 0
        if "fall" in base_name:
            return 1

        return 0

    def discover_video_files(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Discovers videos or extracted frame sequence directories in data_dir.
        Supports:
        1. Extracted RGB image sequence folders (e.g. URFD):
           - data_dir/fall/fall-01-cam0-rgb/*.png
           - data_dir/adl/adl-01-cam0-rgb/*.png
           - data_dir/fall-01-cam0-rgb/*.png (flat structure)
        2. Video files:
           - data_dir/fall/video1.mp4
           - data_dir/adl/video2.avi
        """
        metadata: List[Dict[str, Any]] = []

        if not os.path.exists(data_dir):
            return metadata

        # Normalize path
        data_dir = os.path.abspath(data_dir)

        # Walk directory tree
        for root, dirs, files in os.walk(data_dir):
            # Check if current directory contains image frame sequences
            image_files = [f for f in files if os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS]

            if len(image_files) >= 1:
                # Treat this directory as a single image sequence
                label = self.determine_label(root, data_dir=data_dir)
                subject_id = self.extract_subject_id(root)

                metadata.append({
                    "path": root,
                    "label": label,
                    "subject_id": subject_id,
                    "type": "image_sequence",
                    "filename": os.path.basename(root),
                    "num_frames": len(image_files)
                })
                # Don't recurse into subdirectories if this is already an image sequence folder
                dirs.clear()
                continue

            # Check for standalone video files
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.VIDEO_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    label = self.determine_label(full_path, data_dir=data_dir)
                    subject_id = self.extract_subject_id(full_path)

                    metadata.append({
                        "path": full_path,
                        "label": label,
                        "subject_id": subject_id,
                        "type": "video_file",
                        "filename": file
                    })

        return metadata

    def extract_frames_from_sequence_dir(self, dir_path: str) -> np.ndarray:
        """
        Loads extracted image frames from an image sequence folder (e.g. URFD RGB frames).
        1. Naturally sorts the frame filenames to preserve correct chronological time order.
        2. Downsamples frames from source FPS (default 30 FPS) to target sampling FPS (default 3 FPS).
        3. Resizes frames to (frame_height, frame_width) and converts BGR to RGB uint8.
        """
        all_files = os.listdir(dir_path)
        image_files = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
        ]

        if not image_files:
            return np.empty((0, self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)

        # Natural sort ensures frame_2.png comes before frame_10.png
        image_files.sort(key=natural_sort_key)

        # FPS sampling calculation
        # e.g. source 30 FPS -> target 3 FPS => step = 10 (sample every 10th frame)
        source_fps = self.config.source_fps if self.config.source_fps > 0 else 30.0
        target_fps = self.config.sampling_fps if self.config.sampling_fps > 0 else 3.0
        step = max(1, int(round(source_fps / target_fps)))

        sampled_paths = [os.path.join(dir_path, image_files[i]) for i in range(0, len(image_files), step)]
        sampled_frames = []

        for img_path in sampled_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize to target resolution
            resized = cv2.resize(
                img,
                (self.config.frame_width, self.config.frame_height),
                interpolation=cv2.INTER_AREA
            )
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            sampled_frames.append(rgb_frame)

        if not sampled_frames:
            return np.empty((0, self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)

        return np.array(sampled_frames, dtype=np.uint8)

    def extract_frames_from_video_file(self, video_path: str) -> np.ndarray:
        """
        Extracts frames from a video container (.mp4, .avi, etc.),
        downsampling to target sampling FPS, resized to 224x224 RGB uint8 format.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.empty((0, self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = self.config.source_fps if self.config.source_fps > 0 else 30.0

        step = max(1, int(round(orig_fps / self.config.sampling_fps)))

        sampled_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % step == 0:
                resized = cv2.resize(
                    frame,
                    (self.config.frame_width, self.config.frame_height),
                    interpolation=cv2.INTER_AREA
                )
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                sampled_frames.append(rgb_frame)

            frame_idx += 1

        cap.release()

        if not sampled_frames:
            return np.empty((0, self.config.frame_height, self.config.frame_width, 3), dtype=np.uint8)

        return np.array(sampled_frames, dtype=np.uint8)

    def extract_frames_at_fps(self, path: str) -> np.ndarray:
        """
        Extracts sampled RGB frames from either an image sequence directory or a video file.
        """
        if os.path.isdir(path):
            return self.extract_frames_from_sequence_dir(path)
        else:
            return self.extract_frames_from_video_file(path)

    def extract_sequences_from_frames(
        self,
        frames: np.ndarray,
        label: int,
        stride: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slices downsampled frames into 15-frame temporal sliding window sequences.
        If clip has fewer than 15 frames, repeats the last frame to pad to exactly 15 frames.
        Returns:
            X: float32 array of shape (N, 15, 224, 224, 3) with pixel values [0, 255]
            y: float32 array of shape (N, 1) with label
        """
        seq_len = self.config.sequence_length
        stride = stride if stride is not None else self.config.stride
        total_frames = len(frames)

        if total_frames == 0:
            return (
                np.empty((0, seq_len, self.config.frame_height, self.config.frame_width, 3), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32)
            )

        sequences = []

        if total_frames < seq_len:
            # Video/sequence shorter than 15 sampled frames:
            # Pad by repeating the last frame
            pad_count = seq_len - total_frames
            last_frame = frames[-1:]
            padded = np.concatenate([frames, np.repeat(last_frame, pad_count, axis=0)], axis=0)
            sequences.append(padded)
        else:
            # Sliding window with step = stride
            for start_idx in range(0, total_frames - seq_len + 1, stride):
                window = frames[start_idx: start_idx + seq_len]
                sequences.append(window)

            # Ensure the final tail frames are included
            if (total_frames - seq_len) % stride != 0:
                sequences.append(frames[-seq_len:])

        X = np.array(sequences, dtype=np.float32)
        y = np.full((len(sequences), 1), fill_value=label, dtype=np.float32)
        return X, y

    def extract_sequences_from_video(
        self,
        video_path: str,
        label: int,
        stride: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backwards-compatible interface: extracts 15-frame window sequences
        from a video path (either video file or image sequence directory).
        """
        frames = self.extract_frames_at_fps(video_path)
        return self.extract_sequences_from_frames(frames, label, stride=stride)

    def extract_sequences_from_record(
        self,
        record: Dict[str, Any],
        stride: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts 15-frame window sequences from a discovered dataset record.
        """
        path = record["path"]
        label = int(record["label"])
        return self.extract_sequences_from_video(path, label, stride=stride)

    def load_dataset_split(
        self,
        video_records: List[Dict[str, Any]],
        stride: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and aggregates 15-frame sequences from a list of video records.
        Returns:
            X: array of shape (Total_Sequences, 15, 224, 224, 3)
            y: array of shape (Total_Sequences, 1)
        """
        all_X = []
        all_y = []

        for record in video_records:
            X_v, y_v = self.extract_sequences_from_record(record, stride=stride)
            if len(X_v) > 0:
                all_X.append(X_v)
                all_y.append(y_v)

        if not all_X:
            seq_len = self.config.sequence_length
            h, w, c = self.config.frame_shape
            return np.empty((0, seq_len, h, w, c), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    def create_tf_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 8,
        is_training: bool = True
    ) -> tf.data.Dataset:
        """
        Converts in-memory numpy sequence arrays into an optimized tf.data.Dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if is_training and len(X) > 0:
            dataset = dataset.shuffle(buffer_size=min(len(X), 1000), seed=self.config.random_seed)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
