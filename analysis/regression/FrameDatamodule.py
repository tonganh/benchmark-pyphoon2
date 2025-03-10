import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms.functional import center_crop

from pathlib import Path
import numpy as np
import sys
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
import math


class TyphoonDataModule(pl.LightningDataModule):
    """Typhoon Dataset Module using lightning architecture"""

    def __init__(
        self,
        dataroot,
        batch_size,
        num_workers,
        labels,
        split_by,
        load_data,
        dataset_split,
        standardize_range,
        downsample_size,
        cropped,
        corruption_ceiling_pct=100,
        image_dirs=None,
        metadata_dirs=None,
        metadata_jsons=None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        data_path = Path(dataroot)
        self.images_path = str(data_path / "image") + "/"
        self.track_path = str(data_path / "metadata") + "/"
        self.metadata_path = str(data_path / "metadata.json")
        self.load_data = load_data
        self.split_by = split_by
        self.labels = labels

        self.dataset_split = dataset_split
        self.standardize_range = standardize_range
        self.downsample_size = downsample_size
        self.cropped = cropped

        self.image_dirs = image_dirs
        self.metadata_dirs = metadata_dirs
        self.metadata_jsons = metadata_jsons

        self.corruption_ceiling_pct = corruption_ceiling_pct

    def setup(self, stage):
        # Load Dataset
        dataset = DigitalTyphoonDataset(
            str(self.images_path),
            str(self.track_path),
            str(self.metadata_path),
            self.labels,
            load_data_into_memory=self.load_data,
            filter_func=self.image_filter,
            transform_func=self.transform_func,
            verbose=False,
            image_dirs=self.image_dirs,
            metadata_dirs=self.metadata_dirs,
            metadata_jsons=self.metadata_jsons,
        )
        # generator1 = torch.Generator().manual_seed(3)
        
        if not hasattr(self, 'dataset_split') or not self.dataset_split:
            self.dataset_split = [0.8, 0.2, 0.0]  # default split ratios
        
        # Ensure split ratios sum to 1
        if not math.isclose(sum(self.dataset_split), 1.0):
            print("WARNING: Split ratios don't sum to 1, normalizing...")
            total = sum(self.dataset_split)
            self.dataset_split = [x/total for x in self.dataset_split]
        
        self.train_set, self.val_set, _ = dataset.random_split(
            self.dataset_split, split_by=self.split_by
        )
        
        print(f"Train set size: {len(self.train_set)}")
        print(f"Val set size: {len(self.val_set)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False
        )

    def image_filter(self, image):
        return (
            (image.grade() < 6)
            and (image.grade() > 2)
            and (image.year() != 2023)
            and (100.0 <= image.long() <= 180.0)
        )

    def transform_func(self, image_batch):
        """transform function applied on the images for pre-processing"""
        image_batch = np.clip(
            image_batch, self.standardize_range[0], self.standardize_range[1]
        )
        image_batch = (image_batch - self.standardize_range[0]) / (
            self.standardize_range[1] - self.standardize_range[0]
        )
        if self.downsample_size != (512, 512):
            image_batch = torch.Tensor(image_batch)
            if self.cropped:
                image_batch = center_crop(image_batch, (224, 224))
            else:
                image_batch = torch.reshape(
                    image_batch, [1, 1, image_batch.size()[0],
                                  image_batch.size()[1]]
                )
                image_batch = nn.functional.interpolate(
                    image_batch,
                    size=self.downsample_size,
                    mode="bilinear",
                    align_corners=False,
                )
                image_batch = torch.reshape(
                    image_batch, [image_batch.size()[2], image_batch.size()[3]]
                )
            image_batch = image_batch.numpy()
        return image_batch
