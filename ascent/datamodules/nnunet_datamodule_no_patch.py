import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from lightning import LightningDataModule
from lightning.pytorch.trainer.states import TrainerFn
from monai.data import CacheDataset, DataLoader, IterableDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    SpatialPadd,
)
from sklearn.model_selection import KFold, train_test_split

from ascent import utils
from ascent.preprocessing.preprocessing import resample_image, resample_label
from ascent.utils.data_loading import get_case_identifiers_from_npz_folders
from ascent.utils.dataset import nnUNet_Iterator
from ascent.utils.file_and_folder_operations import load_pickle, save_pickle, subfiles
from ascent.utils.transforms import Convert2Dto3Dd, Convert3Dto2Dd, LoadNpyd, MayBeSqueezed

import pandas as pd
from torch.utils.data import Dataset, random_split
import json
import nibabel as nib
import torchio as tio
from random import shuffle

log = utils.get_pylogger(__name__)


class nnUNetDataset(Dataset):
    def __init__(self, data_path, test_frac=0.1, common_spacing=None, seed=0, test=False, *args,
                 **kwargs):
        super().__init__()
        self.data_path = data_path
        csv_file = self.data_path + '/subset.csv'
        self.df = pd.read_csv(csv_file, index_col=0)
        self.df = self.df[self.df['valid_segmentation'] == True]

        self.common_spacing = common_spacing

        # split according to test_frac
        self.test = test
        test_len = int(test_frac * len(self.df))
        train_val_len = len(self.df) - test_len
        idx_train_val, idx_test = random_split(range(len(self.df)), [train_val_len, test_len])
        if self.test:
            self.df = self.df.iloc[idx_test.indices]
        else:
            self.df = self.df.iloc[idx_train_val.indices]

        print(f"Test step: {test} , len of dataset {len(self.df)}")

        if common_spacing is None:
            self.calculate_common_spacing()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        sub_path = self.get_img_subpath(self.df.iloc[idx])
        img_nifti = nib.load(self.data_path + '/img/' + sub_path)
        img = img_nifti.get_fdata() / 255
        mask = nib.load(self.data_path + '/segmentation/' + sub_path.replace("_0000", "")).get_fdata()
        original_shape = np.asarray(list(img.shape))

        if img.shape[0]*img.shape[1]*img.shape[2] > 10000000:
            time_len = int(10000000 // (img.shape[0]*img.shape[1]))
            start_idx = np.random.randint(low=0, high=img.shape[2]-time_len)
            img = img[..., start_idx:start_idx+time_len]
            mask = mask[..., start_idx:start_idx+time_len]

        # get desired closest divisible shape
        x = round(img.shape[0] // 32) * 32
        y = round(img.shape[1] // 32) * 32
        if not self.test:
            z = round(img.shape[2] // 4) * 4
        else:
            z = img.shape[2]

        if self.common_spacing is not None:
            transform = tio.Resample(self.common_spacing)
            croporpad = tio.CropOrPad((x, y, z))
            img = croporpad(transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=img_nifti.affine))).tensor
            mask = croporpad(transform(tio.LabelMap(tensor=np.expand_dims(mask, 0), affine=img_nifti.affine))).tensor
        else:
            # RESAMPLE NAIVE
            img = torch.tensor(resample_image(np.expand_dims(img, 0), (x, y, z), True, lowres_axis=np.array([2]), verbose=False))
            mask = torch.tensor(resample_label(np.expand_dims(mask, 0), (x, y, z), True, lowres_axis=np.array([2]), verbose=False))

        return {'image': img.type(torch.float32),
                'label': mask.type(torch.float32),
                'image_meta_dict': {'case_identifier': self.df.iloc[idx]['dicom_uuid'],
                                     'original_shape': original_shape,
                                     'original_spacing': img_nifti.header['pixdim'][1:4]}}

    def get_img_subpath(self, row):
        return f"{row['study']}/{row['view'].lower()}/{row['dicom_uuid']}_0000.nii.gz"

    def calculate_common_spacing(self, num_samples=100):
        spacings = np.zeros(3)
        idx = self.df.reset_index().index.to_list()
        shuffle(idx)
        idx = idx[:num_samples]
        for i in idx:
            sub_path = self.get_img_subpath(self.df.iloc[i])
            img_nifti = nib.load(self.data_path + '/img/' + sub_path)
            spacings += img_nifti.header['pixdim'][1:4]
        self.common_spacing = spacings / len(idx)
        print(f"ESTIMATED COMMON AVERAGE SPACING: {self.common_spacing}")


class nnUNetDataModule_no_patch(LightningDataModule):
    """Data module for nnUnet pipeline."""

    def __init__(
        self,
        data_dir: str = "data/",
        dataset_name: str = "CAMUS",
        fold: int = 0,
        batch_size: int = 2,
        patch_size: tuple[int, ...] = (128, 128, 128),
        common_spacing: tuple[float, ...] = None,
        in_channels: int = 1,
        do_dummy_2D_data_aug: bool = True,
        num_workers: int = os.cpu_count() - 1,
        pin_memory: bool = True,
        test_splits: bool = True,
        seg_label: bool = True,
    ):
        """Initialize class instance.

        Args:
            data_dir: Path to the data directory.
            dataset_name: Name of dataset to be used.
            fold: Fold to be used for training, validation or test.
            batch_size: Batch size to be used for training and validation.
            patch_size: Patch size to crop the data..
            in_channels: Number of input channels.
            do_dummy_2D_data_aug: Whether to apply 2D transformation on 3D dataset.
            num_workers: Number of subprocesses to use for data loading.
            pin_memory: Whether to pin memory to GPU.
            test_splits: Whether to split data into train/val/test (0.8/0.1/0.1).
            seg_label: Whether the labels are segmentations.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[torch.utils.Dataset] = None
        self.data_val: Optional[torch.utils.Dataset] = None
        self.data_test: Optional[torch.utils.Dataset] = None

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        More detailed steps:
        1. Split the dataset into train, validation (and test) folds if it was not done.
        2. Use the specified fold for training. Create random 80:10:10 or 80:20 split if requested
           fold is larger than the length of saved splits.
        3. Set variables: `self.data_train`, `self.data_val`, `self.data_test`, `self.data_predict`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if stage == "fit" or stage is None:
            train_set_full = nnUNetDataset(self.hparams.data_dir + '/' + self.hparams.dataset_name, common_spacing=self.hparams.common_spacing)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.data_train, self.data_val = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = nnUNetDataset(self.hparams.data_dir + '/' + self.hparams.dataset_name, test=True, common_spacing=self.hparams.common_spacing)

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=max(self.hparams.num_workers, 1),
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=max(self.hparams.num_workers, 1),
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:  # noqa: D102
        # We use a batch size of 1 for testing as the images have different shapes and we can't
        # stack them

        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    dl = nnUNetDataModule_no_patch('/data/ascent_data/subset_3/', common_spacing=(0.4302, 0.4302, 1.0), dataset_name='', num_workers=1, batch_size=1)

    dl.setup()
    for batch in iter(dl.train_dataloader()):
        print(batch['image'].shape)
        print(batch['label'].shape)

        from matplotlib import pyplot as plt
        plt.imshow(batch['image'][0, 0, :, :, 5])

        plt.figure()
        plt.imshow(batch['label'][0, 0, :, :, 5])
        plt.show()