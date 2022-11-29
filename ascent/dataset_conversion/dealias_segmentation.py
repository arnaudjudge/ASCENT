import os
from pathlib import Path
from typing import Union

import SimpleITK as sitk
from tqdm import tqdm
from utils import generate_dataset_json


def rename_case(
    case_folder: Union[Path, str],
    imagesTr: Union[Path, str],
    labelsTr: Union[Path, str],
    case_identifier: str,
    multiply: bool = False,
) -> None:
    """Rename case filename from Doppler dataset to nnUNet's format.

    Args:
        case_folder: Path to the folder containing the data files of a case.
        imagesTr: Path to the folder to save image file.
        labelsTr: Path to the folder to save label file.
        case_identifier: Case identifier, e.g. BraTS_0001.
        multiply: Whether to multiply Doppler velocity with Doppler power.
    """
    case = os.path.basename(case_folder)
    velocity = sitk.ReadImage(os.path.join(case_folder, f"{case}_3CH.nii.gz"))
    power = sitk.ReadImage(os.path.join(case_folder, f"{case}_3CH_power.nii.gz"))
    gt_seg = sitk.ReadImage(os.path.join(case_folder, f"{case}_3CH_seg.nii.gz"))

    if multiply:
        vel_array = sitk.GetArrayFromImage(velocity)
        power_array = sitk.GetArrayFromImage(power)
        spacing = velocity.GetSpacing()
        mult = vel_array * power_array
        mult_itk = sitk.GetImageFromArray(mult)
        mult_itk.SetSpacing(spacing)
        sitk.WriteImage(mult_itk, os.path.join(imagesTr, f"{case_identifier}_0000.nii.gz"))
    else:
        sitk.WriteImage(velocity, os.path.join(imagesTr, f"{case_identifier}_0000.nii.gz"))
        sitk.WriteImage(power, os.path.join(imagesTr, f"{case_identifier}_0001.nii.gz"))
    sitk.WriteImage(gt_seg, os.path.join(labelsTr, f"{case_identifier}.nii.gz"))


def convert_to_nnUNet(
    data_dir: Union[Path, str],
    dataset_name: Union[Path, str],
    output_dir: Union[Path, str],
    multiply: bool,
) -> None:
    """Convert Doppler dataset to nnUNet's format.

    Args:
        data_dir: Path to the dataset.
        dataset_name: Name of the dataset, e.g. BraTS.
        output_dir: Path to the output folder to save the converted data.
        multiply: Whether to multiply Doppler velocity with Doppler power.
    """
    imagesTr = os.path.join(data_dir, "imagesTr")
    labelsTr = os.path.join(data_dir, "labelsTr")

    id = 1
    for case in tqdm(os.listdir(data_dir)):
        case_identifier = dataset_name + "_%04.0d" % id
        rename_case(os.path.join(data_dir, case), imagesTr, labelsTr, case_identifier, multiply)
        id += 1


if __name__ == "__main__":
    base = "C:/Users/ling/Desktop/Dataset/Doppler/A3C"
    data_dir = "C:/Users/ling/Desktop/Thesis/REPO/ASCENT/data/DEALIASC/raw"
    os.makedirs(data_dir, exist_ok=True)

    dataset_name = "Dealias"

    imagesTr = os.path.join(data_dir, "imagesTr")
    labelsTr = os.path.join(data_dir, "labelsTr")
    imagesTs = os.path.join(data_dir, "imagesTs")
    labelsTs = os.path.join(data_dir, "labelsTs")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    convert_to_nnUNet(base, dataset_name, data_dir, False)
    generate_dataset_json(
        os.path.join(data_dir, "dataset.json"),
        imagesTr,
        imagesTs,
        ("noNorm", "noNorm"),
        {0: "background", 1: "V + 2", 2: "V - 2"},
        dataset_name,
    )