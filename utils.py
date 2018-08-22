#BASIC IMPORTS
import sys

#TORCH IMPORTS
import torchvision.transforms as transforms

#INFERNO IMPORTS
import inferno.io.transform as inftransforms

def get_basic_transforms():
    train_tr = transforms.Compose([ transforms.RandomVerticalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply(
                                            [transforms.RandomAffine(degrees  =180,
                                                                translate=(0.1, 0.1),
                                                                scale    =(0.9, 1.0),
                                                                shear    =10)]),
                                        inftransforms.image.PILImage2NumPyArray(),
                                        inftransforms.image.ElasticTransform(alpha=100, sigma=50),
                                        inftransforms.generic.NormalizeRange(normalize_by=255.0),
                                        inftransforms.generic.AsTorchBatch(dimensionality=2)])

    test_tr  = transforms.Compose([ transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    inftransforms.image.PILImage2NumPyArray(),
                                    inftransforms.generic.NormalizeRange(normalize_by=255.0),
                                    inftransforms.generic.AsTorchBatch(dimensionality=2)])
    return train_tr, test_tr

    
def log_info(message):
    print('INFO: ' + message)
    sys.stdout.flush()