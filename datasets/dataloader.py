import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img
import os
import numpy as np

from albumentations import (
    Compose, HueSaturationValue, HorizontalFlip,
    RandomCrop, VerticalFlip, RandomRotate90, Transpose, ShiftScaleRotate, RandomBrightnessContrast,
    RandomGamma, Blur, OneOf, ElasticTransform, GridDistortion, OpticalDistortion
)


def aug_with_crop(image_size=256, crop_prob=1):
    return Compose([
        RandomCrop(width=image_size, height=image_size, p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        Blur(p=0.1, blur_limit=3),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.4)
    ], p=1)


class TrainAirbusShipSequence(tf.keras.utils.Sequence):
    """
    This class is used to create batch images for training models.

    Parameters
    ----------
    path_to_images : list[str]
        A path to the image folder.
    img_size : tuple[int, int]
        A size of the image. Tenserflow will load images from folder corresponding to this size.
    path_to_masks: list[str]
        A path to the mask folder
    augmentation: None or aug_with_crop
         Apply augmentation to the image and mask, if a function is passed.
    """

    def __init__(self, path_to_images, img_size, path_to_masks, augmentation=None, batch_size=1):
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.img_size = img_size
        self.augmentation = augmentation
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.path_to_images) / self.batch_size))

    def __getitem__(self, idx):
        """
        Method that returns a batch of images
        """
        batch_images_path = self.path_to_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks_path = self.path_to_masks[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        if self.augmentation is not None:
            for i, (image_path, mask_path) in enumerate(zip(batch_images_path, batch_masks_path)):
                img = load_img(image_path, target_size=self.img_size)
                mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
                augmented = self.augmentation(self.img_size[0])(image=np.array(img), mask=np.array(mask))
                x[i] = augmented['image'] / 255
                y[i] = np.expand_dims(augmented['mask'] / 255, 2)
        else:
            for i, (image_path, mask_path) in enumerate(zip(batch_images_path, batch_masks_path)):
                img = load_img(image_path, target_size=self.img_size)
                mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
                x[i] = img
                x = x / 255
                y[i] = np.expand_dims(mask, 2)
                y = y / 255

        return x, y


class TestAirbusShipSequence(tf.keras.utils.Sequence):
    """
    This class is used to create batch images for inference.

    Parameters
    ----------
    path_to_images : list[str]
        A path to the image folder.
    img_size : tuple[int, int]
        A size of the image. Tenserflow will load images from folder corresponding to this size.
    """

    def __init__(self, path_to_images, img_size, batch_size=1):
        self.path_to_images = path_to_images
        self.img_size = img_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.path_to_images) / self.batch_size))

    def __getitem__(self, idx):
        batch_images_path = self.path_to_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")

        for i, image_path in enumerate(batch_images_path):
            img = load_img(image_path, target_size=self.img_size)
            x[i] = img
            x = x / 255
        return x


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


def main():
    path_to_masks = os.listdir(os.path.join(args.mask_folder))
    full_path_to_images = [args.image_folder + i.split('_')[0] + '.jpg' for i in path_to_masks]
    full_path_to_mask = [args.mask_folder + i for i in path_to_masks]
    if args.show_aug is True:
        dataset = TrainAirbusShipSequence(full_path_to_images[:5], (256, 256), full_path_to_mask[:5],
                                          augmentation=aug_with_crop)
    else:
        dataset = TrainAirbusShipSequence(full_path_to_images[:5], (256, 256), full_path_to_mask[:5],
                                          augmentation=None)

    fig, axes = plt.subplots(5, 2, figsize=(28, 20))
    cols = ['Image', 'Target']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    for i, (x, y) in enumerate(dataset):
        axes[i][0] = show_img(x[0], figsize=(28, 20), ax=axes[i][0])
        axes[i][1] = show_img(y[0], figsize=(28, 20), ax=axes[i][1])
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_folder', type=str, help='A path to images folder', required=True)
    parser.add_argument('--mask_folder', type=str, help='A path to  the mask folder, for storing masks', required=True)
    parser.add_argument('--show_aug', type=bool, default=False, help='Show images with/without augmentation')
    args = parser.parse_args()

    main()
