import os
import shutil
import glob
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

# User parameters ##################################################################################

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Change the following to the file path on your system #########
data_root = ''  # this path is atm not used, the image data is loaded from the cropped_image_data_path
cropped_image_data_path = '/home/kavin/ExpertInformedDL_v3/bscan_imgs.p'

# Use None for using the image data only
all_fixation_path_gaze = '/media/16TB_Storage/CenteredData/AMD_Dataset/gaze_tsv_files/'
# all_fixation_path_gaze = None
all_fixation_path_cleaned_response = '/media/16TB_Storage/CenteredData/AMD_Dataset/response_final_cleaned_time_converted/'
# all_fixation_path_cleaned_response = None

# Results directory setup
results_dir = './results'
dt_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
results_dir = os.path.join(results_dir, dt_string+f'_alpha-{0e-2}')
os.makedirs(results_dir)
print(f"Results will be save to {results_dir}")

# Use saved folds for reproducibility
# use_saved_folds = '/data/rishabh/ExpertInformedDL_v3/trial_folds'
use_saved_folds = None  # Set to None to create new folds
n_jobs = 1  # for loading data from disk and z-norming

# Training config
epochs = 20
random_seed = 42
batch_size = 1
folds = 5
test_size = 0.1
val_size = 0.14
l2_weight = 1e-6

# Grid search hyperparameters
alphas = (0.1,)
lrs = (1e-5,)
non_pretrained_lr_scaling = 1e-2
aoi_loss_distance_types = ('cross-entropy',)
model_names = ('vit_small_patch32_224_in21k_subimage',)
depths = (1,)

# Image configuration
image_size = (1055, 703)
patch_size = (32, 32)
gaussian_smear_sigma = 0.5

# End of user parameters #############################################################################

if __name__ == '__main__':
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_saved_folds:
        print(f"---------------Using saved folds from {use_saved_folds}--------------------")
        folds = pickle.load(open(os.path.join(use_saved_folds, 'folds.p'), 'rb'))
        test_dataset = pickle.load(open(os.path.join(use_saved_folds, 'test_dataset.p'), 'rb'))
        image_stats = pickle.load(open(os.path.join(use_saved_folds, 'image_stats.p'), 'rb'))
        test_dataset.compound_label_encoder = pickle.load(open(os.path.join(use_saved_folds, 'compound_label_encoder.p'), 'rb'))
    
    else:
        print("----------------Creating data set-------------------")
        folds, test_dataset, image_stats = get_bscan_test_train_val_folds(
            data_root, image_size=image_size, n_folds=folds, n_jobs=n_jobs,
            cropped_image_data_path=cropped_image_data_path,
            root_drive_path_gaze=all_fixation_path_gaze,
            root_drive_path_cleaned=all_fixation_path_cleaned_response,
            patch_size=patch_size, gaussian_smear_sigma=gaussian_smear_sigma,
            test_size=test_size, val_size=val_size)

        if all_fixation_path_cleaned_response is None:
            print("No gaze data provided, using only the image data")
            save_folds = '/home/kavin/ExpertInformedDL_v3/saved_trial_folds/no_gaze'
        if all_fixation_path_cleaned_response is not None:
            print("Gaze data provided, using both image and gaze data")
            save_folds = '/home/kavin/ExpertInformedDL_v3/saved_trial_folds/gaze'
        
        os.makedirs(save_folds, exist_ok=True)
        print(f"Saving folds to {save_folds}, you may set use_saved_folds to this path to use them in the future")

        pickle.dump(folds, open(os.path.join(save_folds, 'folds.p'), 'wb'))
        pickle.dump(test_dataset, open(os.path.join(save_folds, 'test_dataset.p'), 'wb'))
        pickle.dump(image_stats, open(os.path.join(save_folds, 'image_stats.p'), 'wb'))
        pickle.dump(test_dataset.compound_label_encoder, open(os.path.join(save_folds, 'compound_label_encoder.p'), 'wb'))

    # Ensure no data leakage
    for fold_i, (train_trial_dataset, valid_dataset, train_unique_img_dataset) in enumerate(folds):
        train_names = {x['name'] for x in train_trial_dataset.trial_samples}
        valid_names = {x['name'] for x in valid_dataset.trial_samples}
        train_unique_names = {x['name'] for x in train_unique_img_dataset.trial_samples}
        assert len(valid_names.intersection(train_names)) == 0
        assert len(valid_names.intersection(train_unique_names)) == 0

    # Prepare parameter combinations
    parameters = list(itertools.product(depths, alphas, model_names, lrs, aoi_loss_distance_types))

    # Grid search loop
    for param_i, parameter in enumerate(parameters):
        all_train_loss = all_train_acc = all_valid_loss = all_valid_acc = all_train_f1 = all_val_f1 = 0.0
        fold_cnt = 0

        for fold_i, (train_trial_dataset, valid_dataset, train_unique_img_dataset) in enumerate(folds):
            fold_cnt += 1
            depth, alpha, model_name, lr, aoi_loss_dist = parameter
            model = get_model(model_name, image_size=image_stats['subimage_sizes'], depth=depth, device=device, patch_size=patch_size)
            model_config_string = f"model-{model_name}_alpha-{alpha}_dist-{aoi_loss_dist}_lr-{lr}" + (f'_depth-{model.depth}' if hasattr(model, 'depth') else '')
            print(f"Grid search [{param_i}] of {len(parameters)}: {model_config_string}")

            train_dataset = train_trial_dataset if alpha != 0.0 else train_unique_img_dataset
            class_weights = get_class_weight(train_dataset.labels_encoded, 2).to(device)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(1, epochs // 5), T_mult=1, eta_min=1e-6)

            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs")
                model = nn.DataParallel(model)

            criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_bscan)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_bscan)

            train_loss_list, train_acc_list, valid_loss_list, valid_acc_list, train_f1_list, val_f1_list = train_bscan_model(
                model, f"{model_config_string}_fold_{fold_i}", train_loader, valid_loader,
                results_dir=results_dir, optimizer=optimizer, num_epochs=epochs,
                alpha=alpha, dist=aoi_loss_dist, l2_weight=l2_weight, class_weights=class_weights)

            all_train_loss += sum(train_loss_list) / len(train_loss_list)
            all_train_acc += sum(train_acc_list) / len(train_acc_list)
            all_valid_loss += sum(valid_loss_list) / len(valid_loss_list)
            all_valid_acc += sum(valid_acc_list) / len(valid_acc_list)
            all_train_f1 += sum(train_f1_list) / len(train_f1_list)
            all_val_f1 += sum(val_f1_list) / len(val_f1_list)

        all_train_loss /= fold_cnt
        all_train_acc /= fold_cnt
        all_valid_loss /= fold_cnt
        all_valid_acc /= fold_cnt
        all_train_f1 /= fold_cnt
        all_val_f1 /= fold_cnt

        print(f"Training across folds: Loss={all_train_loss:.4f}, Acc={all_train_acc:.4f}, F1={all_train_f1:.4f}")
        print(f"Validation across folds: Loss={all_valid_loss:.4f}, Acc={all_valid_acc:.4f}, F1={all_val_f1:.4f}")

    # Move log and result images to results directory
    try:
        for file_path in glob.glob('_*00*_bscan_output.log'):
            shutil.move(file_path, results_dir)
    except Exception as e:
        print("Log moving skipped:", e)

    for file_path in glob.glob('*roc_curve.png'):
        shutil.move(file_path, results_dir)