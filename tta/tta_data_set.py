import torch
from torch.utils.data import Dataset
import cv2
import os
import albumentations as A


def readImagFromPath(path):
    bgr_img = cv2.imread(path)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def strong_aug(p=1):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.CLAHE(p=0.5),
    ], p=p)


def readImage(path,times):
    # 读取图片
    origin_image = readImagFromPath(path)
    images = []
    images.append(origin_image)
    for i in range(times):
        images.append(strong_aug()(image=origin_image)["image"])
    return images


class TTA_data_set(Dataset):

    def __init__(self, csv_ddf, root_dir, tta_times):
        self._df = csv_df
        self._root_dir = root_dir
        self.tta_times = tta_times

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        images = readImage(os.path.join(self._root_dir, self._df.iloc[idx, 0] + '.tif'),self.tta_times)
        label = self._df.iloc[idx, 1]
        images_torch = []
        for image in images:
            image = A.Normalize(mean=(0.70244707, 0.54624322, 0.69645334),
                                std=(0.23889325, 0.28209431, 0.21625058))(image=image)["image"]
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            images_torch.append(image)
        return images, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np

    path = "../input/train/test_img.tif"
    images = readImage(path,80)
    print(len(images))
    images = [torch.from_numpy(image).permute(2, 0, 1) for image in images]
    grid_img = torchvision.utils.make_grid(images, nrow=9, pad_value=2)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
