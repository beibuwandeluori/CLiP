import albumentations
from albumentations import *
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import torch
import pandas as pd
import os


def get_transforms(mode_type='train', image_size=512):
    if mode_type == 'train':
        transforms_train = albumentations.Compose([
            albumentations.RandomResizedCrop(image_size, image_size, scale=(0.9, 1), p=1),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
            albumentations.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
            albumentations.CLAHE(clip_limit=(1, 4), p=0.5),
            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.2),
            albumentations.OneOf([
                # albumentations.GaussNoise(var_limit=[10, 50]),
                albumentations.GaussNoise(),
                albumentations.GaussianBlur(),
                albumentations.MotionBlur(),
                albumentations.MedianBlur(),
            ], p=0.2),
            albumentations.Resize(image_size, image_size),
            albumentations.OneOf([
                JpegCompression(),
                Downscale(scale_min=0.1, scale_max=0.15),
            ], p=0.2),
            IAAPiecewiseAffine(p=0.2),
            IAASharpen(p=0.2),
            albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=5, p=0.5),
            albumentations.Normalize(),
        ])
        return transforms_train
    else:
        transforms_valid = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize()
        ])
        return transforms_valid


class RANZERDataset(Dataset):
    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        target_cols = df.iloc[:, 1:12].columns.tolist()
        self.labels = df[target_cols].values
        print('data length:', len(self.labels))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img = cv2.imread(row.file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        label = torch.tensor(self.labels[index]).float()
        if self.mode == 'test':
            return torch.tensor(img).float()
        else:
            return torch.tensor(img).float(), label


if __name__ == '__main__':
    data_dir = '/raid/chenby/CLiP/train'
    df_train = pd.read_csv('/data1/cby/py_project/CLiP/data/csv/train_folds.csv')
    df_train['file_path'] = df_train.StudyInstanceUID.apply(lambda x: os.path.join(data_dir, f'{x}.jpg'))
    dataset = RANZERDataset(df_train, 'train', transform=get_transforms())
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for i, (img, label) in enumerate(train_loader):
        print(i, '/', len(train_loader), img.shape, label.shape, label[0])
        if i == 20:
            break

