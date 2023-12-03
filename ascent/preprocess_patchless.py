import random
import shutil

import nibabel as nib
import torchio as tio
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from bdicardio.utils.viz_utils import show_gif
from vital.utils.image.us.measure import EchoMeasure


if __name__ == "__main__":
    visualize = False
    data_path = "/data/ascent_data/subset_3/"
    out_path = "/data/ascent_data/subset_3_flash/"
    csv_name = 'subset.csv'
    window_len = 7
    df = pd.read_csv(data_path + csv_name, index_col=0)
    df['cut_indexes'] = df.get("cut_indexes", None)

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

        area_diff = np.diff(area_curve)
        diff_min = int(area_diff.argmin())
        min_window = [i for i in range(diff_min-window_len//2, diff_min+window_len//2)]
        diff_max = int(area_diff.argmax())
        max_window = [i for i in range(diff_max - window_len // 2, diff_max + window_len // 2)]

        if random.random() > 0.5:
            slices = [i for i in range(len(area_curve)) if i not in min_window]
        else:
            slices = [i for i in range(len(area_curve)) if i not in max_window]
        print(f"{len(area_curve)} : {[s for s in range(len(area_curve)) if s not in slices]}")
        r['cut_indexes'] = [s for s in range(len(area_curve)) if s not in slices]
        seg_data = seg_data[..., slices]
        img_data = img_data[..., slices]

        if visualize:
            plt.figure()
            plt.plot(area_curve)
            plt.scatter(diff_min, area_curve[diff_min], marker='x')
            plt.scatter(diff_max, area_curve[diff_max], marker='o')
            a = show_gif(seg_data.transpose((2, 1, 0)))
            plt.show()

        p_img = Path(out_path) / 'img' / rel_path
        p_seg = Path(out_path) / 'segmentation' / rel_path.replace("_0000", "")

        p_img.parent.mkdir(parents=True, exist_ok=True)
        p_seg.parent.mkdir(parents=True, exist_ok=True)

        img = nib.Nifti1Image(img_data.astype(np.uint8), img_nifti.affine, img_nifti.header)
        img.to_filename(p_img)
        seg = nib.Nifti1Image(seg_data.astype(np.uint8), seg_nifti.affine, seg_nifti.header)
        seg.to_filename(p_seg)
        df.loc[idx] = r

    df.to_csv(out_path + csv_name)
    #shutil.copy(data_path + csv_name, out_path + csv_name)
