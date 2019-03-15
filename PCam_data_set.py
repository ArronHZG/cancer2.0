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

from PIL import Image


def moramlize(train_path, shuffled_data):
    import tqdm
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
    return image


def croppedImage(image):
    return image


def readImage(path, augmentations=False):
    # 读取图片
    image = readImagFromPath(path)
    # 数据预处理
    image = preTreat(image)

    if augmentations:
        # 数据增强
        image = image[24:72, 24:72, :]  # 中心48中裁剪32
        im_aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
            torchvision.transforms.RandomCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(std=(0.23889325, 0.28209431, 0.21625058),
                                             mean=(0.70244707, 0.54624322, 0.69645334))

        ])
    else:
        image = image[32:64, 32:64, :]
        im_aug = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))
        ])
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    image = im_aug(image)
    image = torch.clamp(image, min=0.0, max=1.0)
    return image


class PCam_data_set(torch.utils.data.Dataset):

    def __init__(self, csv_df, root_dir, usage):
        assert usage == 'train' or usage == 'valid'
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

    path = "input/train/test_img.tif"
    image = readImage(path, augmentations=True)
    batch_tensor = [readImage(path, augmentations=True) for x in range(81)]
    # print(batch_tensor)
    # print(np.array(image).shape)
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=9, pad_value=2)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()

    # csv_url = '../input/train_labels.csv'
    # data = pd.read_csv(csv_url)
    # train_path = '../input/train/'
    # test_path = '../input/test/'
    # data['label'].value_counts()

    # data_set = PCam_data_set(data, train_path, 'train')

    # from sklearn.model_selection import train_test_split
    # tr, vd = train_test_split(data, test_size=0.1, random_state=123)
    # train_set = PCam_data_set(tr, train_path, 'train')
    # valid_set = PCam_data_set(vd, train_path, 'valid')
    # for image, label in valid_set:
    #     print(image.dtype)
    #     print(label)
    #     break
