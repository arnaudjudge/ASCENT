import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from bdicardio.utils.viz_utils import show_gif
from vital.utils.image.us.measure import EchoMeasure


if __name__ == "__main__":
    visualize = False
    data_path = "/home/local/USHERBROOKE/juda2901/dev/data/icardio/subset_7/"
    out_path = "/home/local/USHERBROOKE/juda2901/dev/data/icardio/subset_ES_ED/"
    csv_name = 'subset.csv'
    df = pd.read_csv(data_path + csv_name, index_col=0)
    df = df[df['valid_segmentation']]
    cutoff = 500
    df = df.iloc[:500]

    for idx, r in tqdm(df.iterrows(), total=len(df)):
        r = r.to_dict()
        dicom = r['dicom_uuid']
        rel_path = f"{r['study']}/{r['view'].lower()}/{r['dicom_uuid']}_0000.nii.gz"
        p_img = Path(data_path) / 'img' / rel_path
        p_seg = Path(data_path) / 'segmentation' / rel_path.replace("_0000", "")

        if not p_seg.exists() or not p_img.exists():
            continue

        img_nifti = nib.load(p_img)
        img_data = img_nifti.get_fdata()

        seg_nifti = nib.load(p_seg)
        seg_data = seg_nifti.get_fdata()

        area_curve = EchoMeasure.structure_area(seg_data.transpose((2, 0, 1)), labels=1)
        ED = np.argmax(area_curve)
        ES = np.argmin(area_curve)

        if visualize:
            plt.figure()
            plt.plot(area_curve)
            plt.scatter(ES, area_curve[ES], marker='x')
            plt.scatter(ED, area_curve[ED], marker='o')
            a = show_gif(seg_data.transpose((2, 1, 0)))

            plt.figure()
            plt.imshow(img_data[..., ED].T)
            plt.imshow(seg_data[..., ED].T, alpha=0.3)
            plt.title("ED")
            plt.figure()
            plt.imshow(img_data[..., ES].T)
            plt.imshow(seg_data[..., ES].T, alpha=0.3)
            plt.title("ES")
            plt.show()

        dicom_out = Path(out_path) / dicom
        dicom_out.mkdir(parents=True, exist_ok=True)
        #
        # full sequence
        #
        # img
        p = dicom_out / f'{dicom}_img_sequence.nii.gz'
        img = nib.Nifti1Image(img_data.astype(np.uint8), img_nifti.affine, img_nifti.header)
        img.to_filename(p)

        # mask
        p = dicom_out / f'{dicom}_mask_sequence.nii.gz'
        img = nib.Nifti1Image(seg_data.astype(np.uint8), img_nifti.affine, img_nifti.header)
        img.to_filename(p)

        #
        # ED
        #
        # img
        p = dicom_out / f'{dicom}_img_ED.nii.gz'
        img = nib.Nifti1Image(img_data[..., ED].astype(np.uint8), img_nifti.affine, img_nifti.header)
        img.to_filename(p)

        # mask
        p = dicom_out / f'{dicom}_mask_ED.nii.gz'
        img = nib.Nifti1Image(seg_data[..., ED].astype(np.uint8), img_nifti.affine, img_nifti.header)
        img.to_filename(p)

        #
        # ES
        #
        # img
        p = dicom_out / f'{dicom}_img_ES.nii.gz'
        img = nib.Nifti1Image(img_data[..., ES].astype(np.uint8), img_nifti.affine, img_nifti.header)
        img.to_filename(p)

        # mask
        p = dicom_out / f'{dicom}_mask_ES.nii.gz'
        img = nib.Nifti1Image(seg_data[..., ES].astype(np.uint8), img_nifti.affine, img_nifti.header)
        img.to_filename(p)

    df.to_csv(out_path + csv_name)
