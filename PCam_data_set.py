# 读取图片数据以及csv文件
# 对所有图片做预处理
#    亮度，对比度，饱和度，滤波（未完成）
#    抽取没有信息的太过亮或者暗的图片
# 将数据分为训练数据及验证数据
# 对训练数据做数据增强
#    随机翻转，旋转
#    随机亮度
#    随机对比
#    随机饱和
#    随机中心裁剪为32*32
# 对验证数据和测试数据中心裁剪为32*32
# 对测试数据做TTA

import torch
import torchvision
import cv2
import os
import albumentations as A

from PIL import Image


def noramlize(train_path, shuffled_data):
    from tqdm import tqdm
    import numpy as np
    # As we count the statistics, we can check if there are any completely black or white images
    dark_th = 10 / 255  # If no pixel reaches this threshold, image is considered too dark
    bright_th = 245 / 255  # If no pixel is under this threshold, image is considerd too bright
    too_dark_idx = []
    too_bright_idx = []

    x_tot = np.zeros(3)
    x2_tot = np.zeros(3)
    counted_ones = 0
    for i, idx in tqdm(enumerate(shuffled_data['id']), 'computing statistics...(220025 it total)'):
        path = os.path.join(train_path, idx)
        imagearray = readImagFromPath(path + '.tif')
        # is this too dark
        if imagearray.max() < dark_th:
            too_dark_idx.append(idx)
            continue  # do not include in statistics
        # is this too bright
        if imagearray.min() > bright_th:
            too_bright_idx.append(idx)
            continue  # do not include in statistics
        x_tot += imagearray.mean(axis=0)
        x2_tot += (imagearray ** 2).mean(axis=0)
        counted_ones += 1
    channel_avr = x_tot / counted_ones
    channel_std = np.sqrt(x2_tot / counted_ones - channel_avr ** 2)
    return channel_avr, channel_std


def readImagFromPath(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # # We flip it to rgb for visualization purposes
    # print(path)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def preTreat(image):
    # image = image[32:64, 32:64, :]
    # image = cv2.resize(image, (96, 96), cv2.INTER_LINEAR)
    return image


def strong_aug(p=.5):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
#         Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ], p=p)



def readImage(path, augmentations=False):
    # 读取图片
    image = readImagFromPath(path)
    # 数据预处理
    image = preTreat(image)
    if augmentations:
        # std=(0.23889325, 0.28209431, 0.21625058),
        # mean=(0.70244707, 0.54624322, 0.69645334))
        image = strong_aug(p=1)(image=image)["image"]

    image = A.Normalize(mean=(0.70244707, 0.54624322, 0.69645334),std=(0.23889325, 0.28209431, 0.21625058))(image = image)["image"]
    image=torch.from_numpy(image)
    image=image.permute(2,0,1)
    return image


class PCam_data_set(torch.utils.data.Dataset):

    def __init__(self, csv_df, root_dir, usage):
        assert usage == 'train' or usage == 'valid' or usage == 'test'
        self._augmentations = True if usage == 'train' else False
        self._df = csv_df
        self._root_dir = root_dir


    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        image = readImage(os.path.join(self._root_dir, self._df.iloc[idx, 0] + '.tif'),
                          augmentations=self._augmentations)
        label = self._df.iloc[idx, 1]
        return image, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision
    import numpy as np

    path = "input/train/test_img.tif"
    image = readImage(path, augmentations=True)
    print(image.size())
    print(image)


    batch_tensor = [readImage(path, augmentations=True) for x in range(81)]
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=9, pad_value=2)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
