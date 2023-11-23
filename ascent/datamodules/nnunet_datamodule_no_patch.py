import os
from random import shuffle
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from lightning import LightningDataModule
from monai.data import DataLoader
from torch.utils.data import Dataset, random_split

from ascent import utils

log = utils.get_pylogger(__name__)


class nnUNetDataset(Dataset):
    def __init__(self,
                 data_path,
                 csv_file_name='subset.csv',
                 test_frac=0.1,
                 common_spacing=None,
                 max_window_len=None,
                 use_dataset_fraction=1.0,
                 max_batch_size=None,
                 seed=0,
                 test=False,
                 *args, **kwargs):
        super().__init__()
        self.data_path = data_path
        csv_file = self.data_path + '/' + csv_file_name
        self.df = pd.read_csv(csv_file, index_col=0)
        self.df = self.df[self.df['valid_segmentation'] == True]

        self.max_window_len = max_window_len
        self.max_batch_size = max_batch_size
        if self.max_batch_size and self.max_batch_size > 10:
            print("WARNING: max_batch_size set to a large number, "
                  "behavior is set to use largest batch possible "
                  "if max_batch_size is larger than max calculated length")

        # split according to test_frac
        self.test = test
        test_len = int(test_frac * len(self.df))
        train_val_len = len(self.df) - test_len
        idx_train_val, idx_test = random_split(range(len(self.df)), [train_val_len, test_len])
        if self.test:
            #self.df = self.df.iloc[idx_test.indices]
            self.df = self.df[self.df['dicom_uuid'].isin(['di-049A-A1B8-5410', 'di-07F0-17AE-9F04', 'di-0E79-9C3B-6B19',
                                                         'di-1134-3C8D-029B', 'di-18F1-513A-EDA5', 'di-1A71-4609-FF23',
                                                         'di-27AC-18B7-9FA3', 'di-2FA3-9BFB-17A3', 'di-36B0-4504-F31A',
                                                         'di-42EC-F2BA-C99E', 'di-4948-0A71-457C', 'di-4AB0-241E-442C',
                                                         'di-4DBD-1602-DD41', 'di-51A6-086D-9429', 'di-5619-EC74-BF59',
                                                         'di-6BC1-A03E-3B92', 'di-7824-6815-E07D', 'di-85A5-E58E-6C4E',
                                                         'di-9304-FEEA-BABD', 'di-986D-384A-3BEE', 'di-9E57-DBB2-4313',
                                                         'di-A466-BFB9-2C6F', 'di-AC12-5B26-67C7', 'di-B6BA-69E4-ABAA',
                                                         'di-BA44-0395-AF7A', 'di-BF97-9E80-C01B', 'di-C155-AA91-6271',
                                                         'di-C1D9-A49B-6DEC', 'di-C374-C451-4A49', 'di-C425-2069-8EF8',
                                                         'di-C882-9002-816C', 'di-CCA1-E7EE-7288', 'di-DC43-4169-20E1',
                                                         'di-E572-2EC8-E9C1', 'di-E5B2-B48C-F045', 'di-E74A-291D-150A',
                                                         'di-E8E1-1DE6-D3C5', 'di-EBAE-B9BC-F90B', 'di-EDC8-A514-F22A',
                                                         'di-EEB9-0133-4633', 'di-F2FF-397C-F62E', 'di-F33B-7A20-0BDF',
                                                         'di-F967-7E77-AF69'])]
            print(self.df['dicom_uuid'])
        else:
            self.df = self.df.iloc[idx_train_val.indices]
            self.df = self.df[~self.df['dicom_uuid'].isin(['di-049A-A1B8-5410', 'di-07F0-17AE-9F04', 'di-0E79-9C3B-6B19',
                                                           'di-1134-3C8D-029B', 'di-18F1-513A-EDA5', 'di-1A71-4609-FF23',
                                                           'di-27AC-18B7-9FA3', 'di-2FA3-9BFB-17A3', 'di-36B0-4504-F31A',
                                                           'di-42EC-F2BA-C99E', 'di-4948-0A71-457C', 'di-4AB0-241E-442C',
                                                           'di-4DBD-1602-DD41', 'di-51A6-086D-9429', 'di-5619-EC74-BF59',
                                                           'di-6BC1-A03E-3B92', 'di-7824-6815-E07D', 'di-85A5-E58E-6C4E',
                                                           'di-9304-FEEA-BABD', 'di-986D-384A-3BEE', 'di-9E57-DBB2-4313',
                                                           'di-A466-BFB9-2C6F', 'di-AC12-5B26-67C7', 'di-B6BA-69E4-ABAA',
                                                           'di-BA44-0395-AF7A', 'di-BF97-9E80-C01B', 'di-C155-AA91-6271',
                                                           'di-C1D9-A49B-6DEC', 'di-C374-C451-4A49', 'di-C425-2069-8EF8',
                                                           'di-C882-9002-816C', 'di-CCA1-E7EE-7288', 'di-DC43-4169-20E1',
                                                           'di-E572-2EC8-E9C1', 'di-E5B2-B48C-F045', 'di-E74A-291D-150A',
                                                           'di-E8E1-1DE6-D3C5', 'di-EBAE-B9BC-F90B', 'di-EDC8-A514-F22A',
                                                           'di-EEB9-0133-4633', 'di-F2FF-397C-F62E', 'di-F33B-7A20-0BDF',
                                                           'di-F967-7E77-AF69'])]
            if use_dataset_fraction < 1.0:
                self.df = self.df.sample(frac=use_dataset_fraction)

        print(f"Test step: {test} , len of dataset {len(self.df)}")

        if common_spacing is None:
            self.calculate_common_spacing()
        else:
            self.common_spacing = np.asarray(common_spacing)
            print(f"USING PRESET COMMON SPACING: {self.common_spacing}")

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # Get paths and open images
        sub_path = self.get_img_subpath(self.df.iloc[idx])
        img_nifti = nib.load(self.data_path + '/img/' + sub_path)
        img = img_nifti.get_fdata() / 255
        mask = nib.load(self.data_path + '/segmentation/' + sub_path.replace("_0000", "")).get_fdata()
        original_shape = np.asarray(list(img.shape))

        # limit size of tensor so it can fit on GPU
        if not self.test:
            if img.shape[0] * img.shape[1] * img.shape[2] > 5000000:
                time_len = int(5000000 // (img.shape[0] * img.shape[1]))
                img = img[..., :time_len]
                mask = mask[..., :time_len]

        # transforms and resampling
        if self.common_spacing is None:
            raise Exception("COMMON SPACING IS NONE!")
        transform = tio.Resample(self.common_spacing)
        resampled = transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=img_nifti.affine))

        croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        resampled_cropped = croporpad(resampled)
        resampled_affine = resampled_cropped.affine
        img = resampled_cropped.tensor
        mask = croporpad(transform(tio.LabelMap(tensor=np.expand_dims(mask, 0), affine=img_nifti.affine))).tensor

        if not self.test:
            if self.max_window_len:
                # use partial time window, create as many batches as possible with it unless self.max_batch_size not set
                dynamic_batch_size = img.shape[-1] // self.max_window_len \
                    if not self.max_batch_size or not (self.max_batch_size > 0 and
                                                      (self.max_batch_size * self.max_window_len) < img.shape[-1]) \
                    else self.max_batch_size
                b_img = []
                b_mask = []
                for i in range(dynamic_batch_size):
                    start_idx = np.random.randint(low=0, high=img.shape[-1] - self.max_window_len)
                    b_img += [img[..., start_idx:start_idx + self.max_window_len]]
                    b_mask += [mask[..., start_idx:start_idx + self.max_window_len]]
                img = torch.stack(b_img)
                mask = torch.stack(b_mask)
            else:
                # use entire available time window
                # must unsqueeze to accommodate code in train/val step
                img = img.unsqueeze(0)
                mask = mask.unsqueeze(0)

        return {'image': img.type(torch.float32),
                'label': mask.type(torch.float32),
                'image_meta_dict': {'case_identifier': self.df.iloc[idx]['dicom_uuid'],
                                    'original_shape': original_shape,
                                    'original_spacing': img_nifti.header['pixdim'][1:4],
                                    'original_affine': img_nifti.affine,
                                    'resampled_affine': resampled_affine,
                                    }
                }

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

    def get_desired_size(self, current_shape, divisible_by=(32, 32, 4)):
        # get desired closest divisible bigger shape
        x = int(np.ceil(current_shape[0] / divisible_by[0]) * divisible_by[0])
        y = int(np.ceil(current_shape[1] / divisible_by[1]) * divisible_by[1])
        if not self.test:
            z = int(np.ceil(current_shape[2] / divisible_by[2]) * divisible_by[2])
        else:
            z = current_shape[2]
        return x, y, z


class nnUNetDataModule_no_patch(LightningDataModule):
    """Data module for nnUnet pipeline."""

    def __init__(
        self,
        data_dir: str = "data/",
        dataset_name: str = "CAMUS",
        csv_file_name: str = "subset.csv",
        fold: int = 0,
        batch_size: int = 2,
        patch_size: tuple[int, ...] = (128, 128, 128),
        common_spacing: tuple[float, ...] = None,
        max_window_len: int = None,
        max_batch_size: int = None,
        use_dataset_fraction: float = 1.0,
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
            train_set_full = nnUNetDataset(self.hparams.data_dir + '/' + self.hparams.dataset_name,
                                           csv_file_name=self.hparams.csv_file_name,
                                           common_spacing=self.hparams.common_spacing,
                                           max_window_len=self.hparams.max_window_len,
                                           use_dataset_fraction=self.hparams.use_dataset_fraction,
                                           max_batch_size=self.hparams.max_batch_size)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.data_train, self.data_val = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = nnUNetDataset(self.hparams.data_dir + '/' + self.hparams.dataset_name,
                                           csv_file_name=self.hparams.csv_file_name,
                                           test=True,
                                           common_spacing=self.hparams.common_spacing)

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
    dl = nnUNetDataModule_no_patch('/data/ascent_data/subset_3/',
                                   common_spacing=(0.37, 0.37, 1.0),
                                   max_window_len=4,
                                   max_batch_size=2,
                                   dataset_name='',
                                   num_workers=1,
                                   batch_size=1)

    dl.setup()
    for batch in iter(dl.train_dataloader()):
        bimg = batch['image'].squeeze(0)
        blabel = batch['label'].squeeze(0)
        print(bimg.shape)
        print(blabel.shape)

        from matplotlib import pyplot as plt
        plt.imshow(bimg[0, 0, :, :, 1].T)

        plt.figure()
        plt.imshow(blabel[0, 0, :, :, 1].T)
        plt.show()
