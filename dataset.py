"""
Dataset classes for text-based person re-identification.

Supports annotation JSON with format:
[
    {
        "img_path": "path/to/image.jpg",
        "id": 0,
        "captions": ["caption 1", "caption 2"],
        "captions_bt": ["back-translated 1", "back-translated 2"]  // optional
    }
]

Annotations can be:
  - Split-specific files: data_captions_train.json, data_captions_test.json, etc.
  - A single file with a "split" field per entry.
"""

import json
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from text_utils.tokenizer import tokenize

# CLIP image normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_annotations(anno_dir, split):
    """Load annotations for the given split from anno_dir."""
    # Try split-specific files
    for pattern in [
        f"data_captions_{split}.json",
        f"{split}.json",
        f"{split}_reid.json",
    ]:
        path = os.path.join(anno_dir, pattern)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    # Try combined file with "split" field
    for pattern in ["data_captions.json", "reid_raw.json", "ICFG-PEDES.json", "annotations.json", "data.json"]:
        path = os.path.join(anno_dir, pattern)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return [item for item in data if item.get("split") == split]

    raise FileNotFoundError(
        f"No annotation file found for split '{split}' in {anno_dir}. "
        f"Expected one of: data_captions_{split}.json, {split}.json, "
        f"or a combined file (data_captions.json) with 'split' field."
    )


def _get_img_path_key(entry):
    """Get the image path from an annotation entry (handles different key names)."""
    for key in ("img_path", "file_path", "image_path", "filepath"):
        if key in entry:
            return entry[key]
    raise KeyError(f"No image path key found in annotation entry: {list(entry.keys())}")


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------
class TrainDataset(Dataset):
    """
    Training dataset. Each sample is one (image, caption) pair.
    If an entry has N captions, it produces N samples (same image, different captions).
    """

    def __init__(self, anno_dir, image_dir, transform, back_trans=True):
        annotations = _load_annotations(anno_dir, "train")
        self.image_dir = image_dir
        self.transform = transform
        self.back_trans = back_trans

        self.samples = []
        for entry in annotations:
            img_path = _get_img_path_key(entry)
            pid = entry["id"]
            captions = entry["captions"]
            captions_bt = entry.get("captions_bt", captions)

            for i, cap in enumerate(captions):
                bt = captions_bt[i] if i < len(captions_bt) else captions_bt[-1]
                self.samples.append(
                    {
                        "img_path": img_path,
                        "id": pid,
                        "caption": cap,
                        "caption_bt": bt,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(os.path.join(self.image_dir, s["img_path"])).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "caption": s["caption"],
            "caption_bt": s["caption_bt"] if self.back_trans else s["caption"],
            "id": s["id"],
        }


# ---------------------------------------------------------------------------
# Test datasets (separate image gallery and text queries)
# ---------------------------------------------------------------------------
class TestImageDataset(Dataset):
    """One entry per annotation (one image)."""

    def __init__(self, annotations, image_dir, transform):
        self.entries = []
        for entry in annotations:
            self.entries.append(
                {
                    "img_path": os.path.join(image_dir, _get_img_path_key(entry)),
                    "id": entry["id"],
                }
            )
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        image = Image.open(e["img_path"]).convert("RGB")
        image = self.transform(image)
        return image, e["id"]


class TestTextDataset(Dataset):
    """One entry per caption (an image with N captions produces N entries)."""

    def __init__(self, annotations, context_length=77):
        self.samples = []
        for entry in annotations:
            for cap in entry["captions"]:
                self.samples.append({"caption": cap, "id": entry["id"]})
        self.context_length = context_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens = tokenize([s["caption"]], context_length=self.context_length).squeeze(0)
        return tokens, s["id"]


# ---------------------------------------------------------------------------
# Collate and builder helpers
# ---------------------------------------------------------------------------
def train_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    captions = [b["caption"] for b in batch]
    captions_bt = [b["caption_bt"] for b in batch]
    ids = torch.tensor([b["id"] for b in batch], dtype=torch.long)
    return {"image": images, "caption": captions, "caption_bt": captions_bt, "id": ids}


def build_train_transform(input_resolution):
    h, w = input_resolution
    return transforms.Compose(
        [
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_test_transform(input_resolution):
    h, w = input_resolution
    return transforms.Compose(
        [
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
