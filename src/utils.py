#BASIC IMPORTS
import sys
import numpy as np
from scipy import ndimage
import cv2
from PIL import Image
import math
import random
import scipy.ndimage.interpolation as inter


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

#TORCH IMPORTS
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

#INFERNO IMPORTS
import inferno.io.transform as inftransforms


def image2bag(inp, size=(28, 28), stride=0.5, crop=False, pyramid_layers=1):
    bag=[]
    if crop:
        img, mask = inp[0], inp[1]
        mask  = np.array(mask != mask.min())
        w, h = img.size()
    else:
        img = inp[0]
        w, h = img.size()

    boxes = []
    wsize = size

    while pyramid_layers > 0:
        for cx in range(0, w - wsize[0], int(wsize[0] * stride)):
            for cy in range(0, h - wsize[1], int(wsize[1] * stride)):
                if crop:
                    if mask[cx:cx+wsize[0], cy:cy+wsize[1]].sum() != wsize[0] * wsize[1]:
                        continue
                cropped = img[cx:cx+wsize[0], cy:cy+wsize[1]]
                cropped = local_autoscale_ms(cropped)
                cropped = F.upsample(cropped[None, None, :, :], size=size, mode='bilinear')[0]
                boxes.append((cx, cy, wsize[0], wsize[1]))
                bag.append(cropped)

        ## If bag is empty – repeat everything without cropping
        if len(bag) == 0:
            for cx in range(0, w - wsize[0], int(wsize[0] * stride)):
                for cy in range(0, h - wsize[1], int(wsize[1] * stride)):
                    cropped = img[cx:cx+wsize[0], cy:cy+wsize[1]]
                    cropped = local_autoscale_ms(cropped)[None, :, :]
                    boxes.append((cx, cy, wsize[0], wsize[1]))
                    bag.append(cropped)
                    
        wsize = (int(1.4 * wsize[0]), int(1.4 * wsize[1]))
        pyramid_layers -= 1
    
    return torch.stack(bag), boxes



def local_autoscale(img):
    return np.uint8((img - img.min()) / (img.max() - img.min()) * 255)

def local_autoscale_ms(img):
    return (img - img.mean()) / img.std()



def get_the_central_cell_mask(pil_image, wsize=32, gauss_ker_crop=21, cl_ker=0.02, fe_ker=0.06, debug=1):
    img = np.array(pil_image)
    bin_th = 0.9 * img.mean() / img.max() * 255

    h, w = img.shape
    cl_ker = int(cl_ker * (h + w) / 2)
    fe_ker = int(fe_ker * (h + w) / 2)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    blured = cv2.GaussianBlur(img, (gauss_ker_crop, gauss_ker_crop), 0)
    ret, bins = cv2.threshold(blured, bin_th, 255, cv2.THRESH_BINARY_INV)
    
    close_ker = np.ones((cl_ker, cl_ker),np.uint8)
    bins = cv2.morphologyEx(bins, cv2.MORPH_CLOSE, close_ker)
    fer_ker = np.ones((fe_ker, fe_ker),np.uint8)
    bins = cv2.morphologyEx(bins, cv2.MORPH_ERODE, fer_ker)
    
    filled = ndimage.binary_fill_holes(bins).astype(np.uint8) * 255
    
    filled = cv2.morphologyEx(filled, cv2.MORPH_DILATE, fer_ker)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, 4, cv2.CV_32S)       
    
    centr_num = -1
    for i in range(len(stats[1:])):
        cx, cy, cw, ch, ca = stats[i + 1]
        cenx = cx + cw / 2
        ceny = cy + ch / 2
        if h/2 < cx or h/2 > cx + cw or \
           w/2 < cy or w/2 > cy + ch:
            continue
        if centr_num != -1:
            print('Error: Two centred cells!')
            #return img
        centr_num = i + 1
    
    if centr_num == -1:
        return np.ones((h, w), np.uint8)
    filtered_labels = (labels == centr_num).astype(np.uint8)
    
    closed = filtered_labels
    last_ker = np.ones((wsize, wsize),np.uint8)
    closed = cv2.morphologyEx(closed, cv2.MORPH_DILATE, last_ker)

    # if debug:
    #     yield img
    #     yield bins
    #     yield filled
    #     yield local_autoscale(labels)
    #     yield local_autoscale(closed * img)
    return closed



def get_basic_transforms():
    train_tr = transforms.Compose([ transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomAffine(degrees  =180,
                                                            translate=(0.1, 0.1),
                                                            scale    =(0.9, 1.0)),
                                    inftransforms.image.PILImage2NumPyArray(),
                                    #inftransforms.image.ElasticTransform(alpha=100, sigma=50),
                                    inftransforms.generic.Normalize(),
                                    inftransforms.generic.AsTorchBatch(dimensionality=2)])

    test_tr  = transforms.Compose([ inftransforms.image.PILImage2NumPyArray(),
                                    inftransforms.generic.Normalize(),
                                    inftransforms.generic.AsTorchBatch(dimensionality=2)])
    return train_tr, test_tr

    
def log_info(message):
    print('INFO: ' + message)
    sys.stdout.flush()

def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features



####
## GEN
####

def get_centriolle(h=34, r=6, bg=60, th=1):
    centriolle = np.zeros((bg, bg, bg), dtype=int)
    
    #bottom cup
    for x in range(-r, r):
        yb = int(math.sqrt(r * r - x * x))
        for y in range(-yb, yb):
            centriolle[int(bg/2) + x, int(bg/2) + y, int(bg/2 - h/2)] = 1
    #walls
    for z in range(int(bg/2 - h/2), int(bg/2 + h/2)):
        for x in range(-r, r):
            y = int(math.sqrt(r * r - x * x))
            centriolle[int(bg/2) + x, int(bg/2) + y, z] = 1
            centriolle[int(bg/2) + x, int(bg/2) - y, z] = 1
        for y in range(-r, r):
            x = int(math.sqrt(r * r - y * y))
            centriolle[int(bg/2) + x, int(bg/2) + y, z] = 1
            centriolle[int(bg/2) - x, int(bg/2) + y, z] = 1
            
    #thickness
    for i in range(int((th - 1) / 2)):
        ncentriolle = centriolle.copy()
        for x in range(1, bg-1):
            for y in range(1, bg-1):
                for z in range(1, bg-1):
                    if centriolle[x-1:x+1, y-1:y+1, z-1:z+1].sum() > 0:
                        ncentriolle[x, y, z] = 1
        centriolle = ncentriolle.copy()
    return centriolle

def get_random_projection(shape, depth=10, min_sum=450, sg=5):
    '''
    Args:
        depth:  depth of accumulation
    '''
    gb = int(shape.shape[0] /2)
    angle_1 = random.randint(0, 360)
    angle_2 = random.randint(0, 360)
    angle_3 = random.randint(0, 360)
    shape = inter.rotate(shape, angle_1, axes=(1,2))
    shape = inter.rotate(shape, angle_2, axes=(0,2))
    shape = inter.rotate(shape, angle_3, axes=(0,1))
    h, w, c= shape.shape
    shape = shape[int(h/2) - gb:int(h/2) + gb, int(w/2) - gb:int(w/2) + gb, int(c/2) - gb:int(c/2) + gb]
    
    proj = np.zeros([2*gb, 2*gb])
    while proj.sum() < min_sum:
        height = random.randint(0, shape.shape[2] - depth)
        for x in range(shape.shape[0]):
            for y in range(shape.shape[1]):
                proj[x,y] = shape[x, y, height:min(height + depth, shape.shape[2])].sum()
    return cv2.GaussianBlur(proj,(sg,sg),0)
    
def show_3d_shape(shape):
    z,y,x = shape.nonzero()
    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'red')

def show_2d_slice(sl, size=2):
    fig = plt.figure(figsize=(size, size))
    plt.imshow(sl, interpolation='nearest', cmap='gray')

def weighted_rand(weights):
    value = np.random.random() * np.array(weights).sum()
    ind = 0
    while value > weights[ind]:
        value -= weights[ind]
        ind += 1
    return ind

def get_masked_std(img, mask):
    mask = mask > 0
    return np.nanstd (np.where(mask * img !=0, mask * img ,np.nan))

def check_app_br(img, mask, extr):
    bf = (mask > 0).sum()
    mask = (img > extr * mask) * (mask > 0)
    return mask.sum() == bf

def add_projection(inp, proj, crop=True, stride=0.1, smooth=3, std_th=30, alpha=0.1, debug=False):
    img, mask = inp[:,:,0], inp[:,:,1]
    pil_mask = Image.fromarray(mask)
    mask = np.array(mask != mask.min())
    w, h = img.shape
    
    gmean = np.nanmean(np.where(mask * img !=0, mask * img ,np.nan))
    gstd  = get_masked_std(img, mask)
    g5per = np.percentile(img, 2)
    
    boxes = []
    wsize = proj.shape

    add_one = False
    for cx in range(0, w - wsize[0], int(wsize[0] * stride)):
        for cy in range(0, h - wsize[1], int(wsize[1] * stride)):
            if crop and mask[cx:cx+wsize[0], cy:cy+wsize[1]].sum() != wsize[0] * wsize[1]:
                continue
#             if  img[cx:cx+wsize[0], cy:cy+wsize[1]].mean() < cent_color
#                 continue
            boxes.append((cx, cy, wsize[0], wsize[1]))

    if len(boxes) == 0:
        return img, 0
      
    weights = [1 / (float(img[cx:cx+wx, cy:cy+wy].std()) ** 6) if check_app_br(img[cx:cx+wx, cy:cy+wy], proj, gstd * alpha) else 0 for cx, cy, wx, wy in boxes]
    
#     if debug:
#         timg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         for i in range(100):
#             cx, cy, wx, wy = boxes[weighted_rand(weights)]
#             color = (255, 0, 0)
#             alpha = 0.05
#             timg = cv2.addWeighted(timg, 1.0 - alpha, cv2.rectangle(timg.copy(), (cy,cx), (cy+wy, cx+wx), color), alpha, 0.0)
#         show_2d_slice(timg, 12)
        
    cx, cy, wx, wy = boxes[weighted_rand(weights)]
    if debug:
        color = (255, 0, 0)
        alpha = 0.5
        img = cv2.addWeighted(img, 1.0 - alpha, cv2.rectangle(img.copy(), (cy,cx), (cy+wy, cx+wx), color), alpha, 0.0)

    for x in range(wx):
        for y in range(wy):
            img[cx + x, cy + y] = max(g5per, img[cx + x, cy + y] - gstd * alpha * proj[x, y] * (np.random.random() / 5 + 0.9))
#             if proj[x, y] > 0:
#                 img[cx + x, cy + y] = max(0, int(gmean * (1 - gstd * proj[x, y] / 255 * (1 - alpha))))

    pil_img  = Image.fromarray(img)
    rot_mask = Image.new('L', (w, h), (1))
    ret_img  = Image.merge("RGB", [pil_img, pil_mask, rot_mask])
    return ret_img, 1