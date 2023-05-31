import numpy as np
import pandas as pd
import os
import concurrent.futures
from PIL import Image


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def get_mask(im, gb, path):
    '''
    get_mask: A function for saving the masks of ships
    '''
    rle = gb.get_group(im).EncodedPixels
    im_mask = masks_as_image(rle)

    im_mask = im_mask * 255
    out_mask = np.stack([im_mask, im_mask, im_mask], axis=2)

    out_im = Image.fromarray(out_mask)

    fn = f'{im[:-4]}_mask.png'
    out_im.save(path + fn, "PNG")


def multi_masks(filenames, gb, path):
    with concurrent.futures.ThreadPoolExecutor(8) as e:
        e.map(lambda x: get_mask(x, gb, path), filenames)

def main():

    if not os.path.exists(args.mask_folder):
        os.makedirs(args.mask_folder)

    # Load competition data
    seg_df = pd.read_csv(args.path_csv)
    # Count the number of ships on images
    seg_df['ships'] = seg_df['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    # add columns with label if the image has ship or not
    unique_img_ids = seg_df.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
    # Separate images (path to the image) that have ship and not
    ship_files = unique_img_ids[unique_img_ids.has_ship > 0].ImageId.values
    empty_files = unique_img_ids[unique_img_ids.has_ship == 0].ImageId.values

    samp_df = seg_df[seg_df.ImageId.isin(ship_files)].copy()

    samp_df.reset_index(inplace=True)

    samp_unique = unique_img_ids[unique_img_ids.ImageId.isin(ship_files)]

    enc_group = samp_df.groupby('ImageId')
    image_fns = samp_unique.ImageId.values
    print("Start create mask")
    multi_masks(image_fns, enc_group, args.mask_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--path_csv', type=str, help='A path to csv file', required=True)
    parser.add_argument('--mask_folder', type=str, help='A path to  the mask folder, for storing masks', required=True)
    args = parser.parse_args()

    main()
