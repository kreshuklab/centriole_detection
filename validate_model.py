import numpy as np
import src.implemented_models as impl_models
import torch.nn.functional as F
import torch
from src.datasets import CentriollesDatasetPatients
from src.utils import log_info, get_basic_transforms
from src.utils import init_weights
import src.implemented_models as impl_models
from inferno.trainers.basic import Trainer
from tqdm import tqdm

model = impl_models.MIL_DenseNet_3fc_bn
path_to_model_weights = '../centrioles/models/MIL_DenseNet_3fc_bn/rerun/weights'
trainer = Trainer(model)
if torch.cuda.is_available():
    trainer = trainer.load(from_directory=path_to_model_weights,
                           best=True)
else:
    trainer = trainer.load(from_directory=path_to_model_weights, 
                           filename='best_checkpoint.pytorch',
                           best=True, map_location='cpu')
model = trainer.model

train_tr, test_tr = get_basic_transforms()
dataset = CentriollesDatasetPatients(nums=[396, 397, 401, 409, 40311, 40318, 40918, 406180, 406183],
                                     main_dir='../centrioles/dataset/new_edition/new_data_png', 
                                     all_data=True, transform=test_tr, inp_size=512)

false_positives = []
false_negatives = []
true_resp = 0

for ind, (img, label) in tqdm(enumerate(dataset)):
    img = img[None, :, :, :]
    responce = list(F.sigmoid(model(img))[0])
    responce = np.argmax(responce)

    if responce != label:
        print(responce)
        if label == 1:
            false_positives.append(ind)
        if label == 0:
            false_negatives.append(ind)
    else:
        true_resp += 1

accuracy = true_resp / len(dataset)
print(accuracy)

with open('result.txt', 'w') as output:
    print(accuracy, file=output)

with open('false_positives.txt', 'w') as output:
    for label in false_positives:
        print(dataset.path[label], file=output)
with open('false_negatives.txt', 'w') as output:
    for label in false_negatives:
        print(dataset.path[label], file=output)
