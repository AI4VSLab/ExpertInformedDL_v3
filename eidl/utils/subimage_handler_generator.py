
import itertools
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from eidl.datasets.BscanDataset import get_bscan_test_train_val_folds
from eidl.datasets.OCTDataset import get_oct_test_train_val_folds
from eidl.utils.iter_utils import collate_fn
from eidl.utils.model_utils import get_model, get_subimage_model2
from eidl.utils.training_utils import train_oct_model, get_class_weight, train_bscan_model


get_subimage_model2()
