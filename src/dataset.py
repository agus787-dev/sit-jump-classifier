"""Dataset and DataLoader code.

One sample = one folder of ordered frames (one video clip).
    sample tensor shape: [T, C, H, W]      e.g. [12, 3, 128, 128]
    batch tensor shape:  [B, T, C, H, W]   e.g. [2, 12, 3, 128, 128]
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, MEAN, STD, TEST_DIR, TRAIN_DIR, VAL_DIR


def get_frame_transform():
    """Transform applied to every single frame (same for train and eval)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def flip_sequence_horizontally(sequence: torch.Tensor) -> torch.Tensor:
    """Mirror all frames of a clip left-right (last dimension is the width)."""
    return torch.flip(sequence, dims=[-1])


class VideoSequenceDataset(Dataset):
    """Reads clips stored as <root_dir>/<class_name>/<clip_folder>/frame_XX.jpg."""

    def __init__(self, root_dir: Path, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.frame_transform = get_frame_transform()

        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} not found. Run prepare_sequences.py first.")

        self.class_names = self._find_class_names()
        self.samples = self._find_samples()   # list of (sequence_dir, label)

    def _find_class_names(self):
        """Every sub-folder of root_dir is one class."""
        return sorted(p.name for p in self.root_dir.iterdir() if p.is_dir())

    def _find_samples(self):
        """Collect (clip_folder, label_index) for every clip."""
        class_to_index = {name: index for index, name in enumerate(self.class_names)}
        samples = []
        for class_name in self.class_names:
            for sequence_dir in sorted((self.root_dir / class_name).iterdir()):
                if sequence_dir.is_dir():
                    samples.append((sequence_dir, class_to_index[class_name]))
        return samples

    def _load_sequence(self, sequence_dir: Path) -> torch.Tensor:
        """Load all frames of one clip into a tensor [T, C, H, W]."""
        frame_paths = sorted(sequence_dir.glob("*.jpg"))
        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            frames.append(self.frame_transform(image))
        return torch.stack(frames, dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence_dir, label = self.samples[index]
        sequence = self._load_sequence(sequence_dir)

        # Flip is decided ONCE per clip, so all frames are flipped together
        # and the motion stays consistent.
        if self.augment and random.random() < 0.5:
            sequence = flip_sequence_horizontally(sequence)

        return sequence, label


def create_dataloaders(batch_size: int = 2):
    """Build train / val / test loaders. Returns them together with class names."""
    train_dataset = VideoSequenceDataset(TRAIN_DIR, augment=True)
    val_dataset = VideoSequenceDataset(VAL_DIR, augment=False)
    test_dataset = VideoSequenceDataset(TEST_DIR, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.class_names


def print_batch_info(loader: DataLoader, class_names):
    """Debug helper: print the shapes of one batch."""
    sequences, labels = next(iter(loader))
    print("Class names:", class_names)
    print("Batch shape [B, T, C, H, W] =", tuple(sequences.shape))
    print("One sample shape [T, C, H, W] =", tuple(sequences[0].shape))
    print("Labels shape:", tuple(labels.shape))
