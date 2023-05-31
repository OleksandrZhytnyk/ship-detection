from tensorflow import keras
import os
import numpy as np
from datasets.dataloader import TrainAirbusShipSequence, TestAirbusShipSequence, show_img
import matplotlib.pyplot as plt


def segment_images_with_mask(model, path_to_images, path_to_mask, image_size):
    dataset = TrainAirbusShipSequence(path_to_images, image_size, path_to_mask, augmentation=None)

    output = model.predict(dataset)
    output = np.where(output > args.threshold, 1, 0)
    return output, dataset


def segment_images_without_mask(model, path_to_images, image_size):
    dataset = TestAirbusShipSequence(path_to_images, image_size,)

    output = model.predict(dataset)
    output = np.where(output > args.threshold, 1, 0)
    return output, dataset


def display_segmentation_with_mask(predicted_mask, dataset):
    length = predicted_mask.shape[0]
    print(length)
    fig, axes = plt.subplots(5, 3, figsize=(28, 20))
    cols = ['Image', 'Target', 'Predicted_mask']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    for i, (x, y) in enumerate(dataset):
        axes[i][0] = show_img(x[0], figsize=(28, 20), ax=axes[i][0])
        axes[i][1] = show_img(y[0], figsize=(28, 20), ax=axes[i][1])
        axes[i][2] = show_img(predicted_mask[i], figsize=(28, 20), ax=axes[i][2])
    plt.show()


def display_segmentation_without_mask(predicted_mask, dataset):
    length = predicted_mask.shape[0]
    print(length)
    fig, axes = plt.subplots(5, 2, figsize=(28, 20))
    cols = ['Image', 'Predicted_mask']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
    for i, (x) in enumerate(dataset):
        axes[i][0] = show_img(x[0], figsize=(28, 20), ax=axes[i][0])
        axes[i][1] = show_img(predicted_mask[i], figsize=(28, 20), ax=axes[i][1])
    plt.show()


def main():

    model = keras.models.load_model(args.checkpoint, compile=False)

    path_to_masks = os.listdir(os.path.join(args.mask_folder))

    full_path_to_images = [args.image_folder + i.split('_')[0] + '.jpg' for i in path_to_masks]
    full_path_to_mask = [args.mask_folder + i for i in path_to_masks]

    if args.test_images is False:
        predicted_mask, dataset = segment_images_with_mask(model, full_path_to_images[-5:], full_path_to_mask[-5:],
                                                           (args.image_size, args.image_size))
        display_segmentation_with_mask(predicted_mask, dataset)
    else:
        predicted_mask, dataset = segment_images_without_mask(model, full_path_to_images[-5:], (args.image_size, args.image_size))
        display_segmentation_without_mask(predicted_mask, dataset)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, help='A path to checkpoint', required=True)
    parser.add_argument('--image_folder', type=str, help='A path to images folder', required=True)
    parser.add_argument('--mask_folder', type=str, help='A path to masks folder', required=True)
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for training and dataloader')
    parser.add_argument('--threshold', type=float, default=0.5, help='A threshold for the predicted mask')
    parser.add_argument('--test_images', type=bool, default=False, help='Predict on test images without mask')
    args = parser.parse_args()

    main()
