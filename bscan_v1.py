import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import itertools
import pickle
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from eidl.datasets.BscanDataset import get_bscan_test_train_val_folds
from eidl.utils.iter_utils import collate_fn, collate_fn_bscan
from eidl.utils.model_utils import get_model, get_subimage_model2
from eidl.utils.training_utils import train_oct_model, get_class_weight, train_bscan_model


# get_subimage_model2()
# User parameters ##################################################################################

# Change the following to the file path on your system #########
data_root = ''  # this path is atm not used, the image data is loaded from the cropped_image_data_path

#torch.autograd.set_detect_anomaly(True)
cropped_image_data_path = '/data/kuang/David/ExpertInformedDL_v3/bscan_v2.p'  # this file is loaded in BscanDataset.get_bscan_data
all_karen_tsv_fixation_path = '/data/leo/data/BScan/ExpertEyetracking/all_karen.tsv'  # this file is used to fix the fixation points

results_dir = '/data/leo/temp/bscan'
dt_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
results_dir = os.path.join(results_dir, dt_string)
os.mkdir(results_dir)
print(f"Results will be save to {results_dir}")

use_saved_folds = '/data/leo/temp/bscan/vit'  # set this to a path to use the saved folds, set to None to create new folds

n_jobs = 5  # n jobs for loading data from hard drive and z-norming the subimages

# generic training parameters ##################################
epochs = 100
random_seed = 42
# IMPORTANT: this must be one for BScan at present, because some of the images doesn't have fixation data.
# So we don't have their AOIs.
# When creating a batch, if some images have AOIs and some don't, we can't concatenate them to create a single tensor.
# So we have to set batch_size to 1.
batch_size = 1
folds = 10

test_size = 0.1
val_size = 0.14

l2_weight = 1e-6

# grid search hyper-parameters ##################################

# depths = 1, 3
# depths = 12,
depths = 1,

################################################################
# alphas = 0.0, 1e-2, 0.1, 0.25, 0.5, 0.75, 1.0
# alphas = 0., 1e-2, 0.1, 0.5
alphas = 1e-2,

################################################################
# lrs = 1e-2, 1e-3, 1e-4
# lrs = 1e-4, 1e-5
# lrs = 1e-4,
lrs = 1e-4,

non_pretrained_lr_scaling = 1e-2

################################################################
# aoi_loss_distance_types = 'Wasserstein', 'cross-entropy'
aoi_loss_distance_types = 'cross-entropy',

################################################################
# model_names = 'base', 'vit_small_patch32_224_in21k', 'vit_small_patch16_224_in21k', 'vit_large_patch16_224_in21k'
# model_names = 'base', 'vit_small_patch32_224_in21k'
#model_names = 'vit_small_patch16_224_in21k_subimage',
model_names = 'vit_small_patch32_224_in21k_subimage',
# model_names = 'base_subimage',
# model_names = 'inception_v4_subimage',
# model_names = 'resnet50_subimage',
# model_names = 'vgg19_subimage',

# grid_search_params = {
#     'vit_small_patch32_224_in21k_subimage': {
#         'alphas': 1e-2,
#         'lrs': 1e-3,
#         'aoi_loss_distance_types': 'cross-entropy',
#         'optimizer': optim.SGD,
#     },
#
#     'base_subimage': {
#         'alphas': 1e-2,
#         'lrs': 1e-4,
#         'aoi_loss_distance_types': 'cross-entropy',
#         'depths': 1,
#         'optimizer': optim.Adam,},
#
#     'inception_v4_subimage': {
#         'lrs': 1e-3,
#         'optimizer': optim.SGD,
#     }
# }

################################################################
image_size = 5275, 703
#patch_size = 16, 16
patch_size = 32, 32
gaussian_smear_sigma = 0.5

# end of user parameters #############################################################################
if __name__ == '__main__':

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_saved_folds:
        print(f"Using saved folds from {use_saved_folds}")
        folds = pickle.load(open(os.path.join(use_saved_folds, 'folds.p'), 'rb'))
        test_dataset = pickle.load(open(os.path.join(use_saved_folds, 'test_dataset.p'), 'rb'))
        image_stats = pickle.load(open(os.path.join(use_saved_folds, 'image_stats.p'), 'rb'))
        test_dataset.compound_label_encoder = pickle.load(open(os.path.join(use_saved_folds, 'compound_label_encoder.p'), 'rb'))
    else:
        print("Creating data set")
        folds, test_dataset, image_stats = get_bscan_test_train_val_folds(data_root, image_size=image_size, n_folds=folds, n_jobs=n_jobs,
                                                                                    cropped_image_data_path=cropped_image_data_path,
                                                                                    all_karen_tsv_fixation_path=all_karen_tsv_fixation_path,
                                                                                    patch_size=patch_size, gaussian_smear_sigma=gaussian_smear_sigma,
                                                                                    test_size=test_size, val_size=val_size)
        print(f"Saving folds to {results_dir}, you may set use_saved_folds to this path to use them in the future")
        pickle.dump(folds, open(os.path.join(results_dir, 'folds.p'), 'wb'))
        pickle.dump(test_dataset, open(os.path.join(results_dir, 'test_dataset.p'), 'wb'))
        pickle.dump(image_stats, open(os.path.join(results_dir, 'image_stats.p'), 'wb'))
        pickle.dump(test_dataset.compound_label_encoder, open(os.path.join(results_dir, 'compound_label_encoder.p'), 'wb'))

    # check there's no data leak between the train and valid in the folds
    for fold_i, (train_trial_dataset, valid_dataset, train_unique_img_dataset) in enumerate(folds):
        train_names = {x['name'] for x in train_trial_dataset.trial_samples}
        valid_names = {x['name'] for x in valid_dataset.trial_samples}

        train_unique_names = {x['name'] for x in train_unique_img_dataset.trial_samples}

        assert len(valid_names.intersection(train_names)) == 0
        assert len(valid_names.intersection(train_unique_names)) == 0

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # TODO use the test loader in the future

    # process the grid search parameters
    parameters = set()
    for depth, alpha, model_name, lr, aoi_loss_dist in itertools.product(depths, alphas, model_names, lrs, aoi_loss_distance_types):
        print(model_name)
        this_lr = lr
        this_depth = depth
        if 'inception' in model_name or 'resnet' in model_name or 'vgg' in model_name:  # inception net doesn't have depth and alpha
            this_params = (model_name, None, None, None, this_lr)
        else:
            this_params = (model_name, this_depth, alpha, aoi_loss_dist, this_lr)
        parameters.add(this_params)

    for param_i, parameter in enumerate(parameters):  # iterate over the grid search parameters
        for fold_i, (train_trial_dataset, valid_dataset, train_unique_img_dataset) in enumerate(folds):
            model_name, depth, alpha, aoi_loss_dist, lr = parameter
            model = get_model(model_name, image_size=image_stats['subimage_sizes'], depth=depth, device=device, patch_size=patch_size)
            print(model_name)
            model_config_string = f"model-{model_name}_alpha-{alpha}_dist-{aoi_loss_dist}_lr-{lr}" + (f'_depth-{model.depth}' if hasattr(model, 'depth') else '')
            print(f"Grid search [{param_i}] of {len(parameters)}: {model_config_string}")

            if 'inception' in model_name or 'resnet' in model_name or 'vgg' in model_name or alpha == 0.0:
                train_dataset = train_unique_img_dataset
            else:
                train_dataset = train_trial_dataset
            # train_dataset.create_aoi(use_subimages=True)
            # valid_dataset.create_aoi(use_subimages=True)

            class_weights = get_class_weight(train_dataset.labels_encoded, 2).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            # optimizer = optim.SGD(model.parameters(), lr=lr)

            if epochs > 1:
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs // 5, T_mult=1, eta_min=1e-6, last_epoch=-1)
            else:
                scheduler = None

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # Wrap the model with nn.DataParallel
                model = nn.DataParallel(model)

            criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_bscan)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_bscan)

            train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = train_bscan_model(
                model, f"{model_config_string}_fold_{fold_i}", train_loader, valid_loader, results_dir=results_dir, optimizer=optimizer, num_epochs=epochs,
                alpha=alpha, dist=aoi_loss_dist, l2_weight=l2_weight, class_weights=class_weights)

    # viz_oct_results(results_dir, test_image_path, test_image_main, batch_size, image_size, n_jobs=n_jobs)

    
#results_dir, batch_size, n_jobs=1, acc_min=.3, acc_max=1, viz_val_acc=True, plot_format='individual', num_plot=14,
    #   rollout_transparency=0.75, figure_dir=None