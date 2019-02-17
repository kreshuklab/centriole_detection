import mrcfile
import cv2
from tqdm import tqdm
import numpy as np
import os


def local_autoscale(img):
    return np.uint8((img - img.min()) / (img.max() - img.min()) * 255)


def tif_and_mrc_to_png(dir_name, out_dir):
    img_names = [f for f in os.listdir(dir_name) if f.endswith('.mrc') or f.endswith('.tif')]
    for img_name in tqdm(img_names):
        img_full_name = os.path.join(dir_name, img_name)
        out_name = os.path.join(out_dir, img_name[:-4] + '.png')
        if img_name[-4:] == '.mrc':
            with mrcfile.open(img_full_name) as img:
                if len(img.data.shape) == 3:
                    print('ERROR: ' + dir_name + img_name + ' is not just one slice!')
                    continue
                out_img = img.data
        if img_name[-4:] == '.tif':
            out_img = cv2.imread(img_full_name)
        out_img = local_autoscale(out_img)
        cv2.imwrite(out_name, out_img)
