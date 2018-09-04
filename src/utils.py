#BASIC IMPORTS
import sys

#TORCH IMPORTS
import torch
import torchvision.transforms as transforms

#INFERNO IMPORTS
import inferno.io.transform as inftransforms



def image2bag(inp, wsize=(28, 28), stride=0.5, crop=False):
    bag=[]
    if crop:
        img, mask, rmask = inp
    else:
        img = inp

    c, w, h = img.size()

    for cx in range(0, w - wsize[0], int(wsize[0] * stride)):
        for cy in range(0, h - wsize[1], int(wsize[1] * stride)):
            if crop:
                if mask[:,cx:cx+wsize[0], cy:cy+wsize[1]].sum() == 0:
                    continue
                if rmask[:,cx:cx+wsize[0], cy:cy+wsize[1]].sum() < wsize[0] * wsize[1]:
                    continue
            cropped = img[:,cx:cx+wsize[0], cy:cy+wsize[1]]
            bag.append(cropped)

    return torch.stack(bag)

def get_the_central_cell_mask(pil_image, gauss_ker_crop=21, cl_ker=10, fe_ker=30, se_ker=400, debug=1):
    img = numpy.array(pil_image) 
    img = img[:, :, ::-1].copy() 
    bin_th = 0.9 * img.mean() / img.max() * 255

    img = global_autoscale(img)
    h, w = img.shape
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    blured = cv2.GaussianBlur(img, (gauss_ker_crop, gauss_ker_crop), 0)
    ret, bins = cv2.threshold(blured, bin_th, 255, cv2.THRESH_BINARY_INV)
    
    close_ker = np.ones((cl_ker, cl_ker),np.uint8)
    bins = cv2.morphologyEx(bins, cv2.MORPH_CLOSE, close_ker)
    fer_ker = np.ones((fe_ker, fe_ker),np.uint8)
    bins = cv2.morphologyEx(bins, cv2.MORPH_ERODE, fer_ker)
    
    filled = ndimage.binary_fill_holes(bins).astype(np.uint8) * 255
    
    filled = cv2.morphologyEx(filled, cv2.MORPH_DILATE, fer_ker)
    
    ser_ker = np.ones((se_ker, se_ker),np.uint8)
    filled = cv2.morphologyEx(filled, cv2.MORPH_ERODE, ser_ker)
    
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
            return img
        centr_num = i + 1
    
    if centr_num == -1:
        print('Error: Could not find cell in da center')

        return img
    filtered_labels = (labels == centr_num).astype(np.uint8)
    
    closed = cv2.morphologyEx(filtered_labels, cv2.MORPH_DILATE, ser_ker)
    closed = cv2.morphologyEx(closed, cv2.MORPH_DILATE, fer_ker)
    
#     if debug:
#         yield bins
#         yield filled
#         yield local_autoscale(labels)
#         yield local_autoscale(img * closed)
    #return img * closed
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