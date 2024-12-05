import os
import pickle
import re
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

from eidl.utils.bscan_expert_fixations import load_all_karen_fixations, load_all_fixations
from eidl.utils.image_utils import generate_image_binary_mask, resize_image, load_bscan_image, get_heatmap
from eidl.utils.SubimageHandler import SubimageHandler
import itertools
import pickle
from datetime import datetime

from torch import optim, nn
from torch.utils.data import DataLoader

from eidl.datasets.BscanDataset import get_bscan_test_train_val_folds
from eidl.utils.iter_utils import collate_fn, collate_fn_bscan
from eidl.utils.model_utils import get_model, get_subimage_model2
from eidl.utils.training_utils import train_oct_model, get_class_weight, train_bscan_model

data_root = ''  # this path is atm not used, the image data is loaded from the cropped_image_data_path

#torch.autograd.set_detect_anomaly(True)
cropped_image_data_path = '/data/kuang/David/ExpertInformedDL_v3/bscan_v2.p'  # this file is loaded in BscanDataset.get_bscan_data
all_karen_tsv_fixation_path = '/data/leo/data/BScan/ExpertEyetracking/all_karen.tsv'  # this file is used to fix the fixation points
all_fixation_path_gaze = '/data/rishabh/ExpertInformedDL_v3/Gaze/'
all_fixation_path_cleaned_response = '/data/rishabh/ExpertInformedDL_v3/cleaned_time_converted/'

use_saved_folds = None #'/data/rishabh/ExpertInformedDL_v3/folds' # '/data/leo/temp/bscan/vit'  # set this to a path to use the saved folds, set to None to create new folds

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
depths = 1
alphas = 1e-2
lrs = 1e-4

non_pretrained_lr_scaling = 1e-2
aoi_loss_distance_types = 'cross-entropy'
model_names = 'vit_small_patch32_224_in21k_subimage'
image_size = 5275, 703
patch_size = 32, 32
gaussian_smear_sigma = 0.5


def get_bscan_test_train_val_folds(data_root, image_size, n_folds, test_size=0.1, val_size=0.1, n_jobs=1, random_seed=None, *args, **kwargs):
    trial_samples, name_label_images_dict, image_labels, image_stats = get_bscan_data(data_root, image_size, n_jobs, *args, **kwargs)
    return trial_samples

def get_bscan_data(data_root, image_size, n_jobs=1, cropped_image_data_path=None, all_karen_tsv_fixation_path=None, root_drive_path_gaze=None, root_drive_path_cleaned=None, patch_size=(32, 32), *args, **kwargs):
    pvalovia_dir = os.path.join(data_root, 'pvalovia-data')

    subimage_loader = SubimageHandler()
    subimage_data = pickle.load(open(cropped_image_data_path, 'rb'))
    image_data = subimage_loader.load_image_data(subimage_data, n_jobs=n_jobs, *args, **kwargs)
    load_image_args = [(image_name, image_size, image_info_dict['original_image']) for image_name, image_info_dict in subimage_data.items()]
    with Pool(n_jobs) as p:
        image_data_dict = dict(p.starmap(resize_image, load_image_args))  # this dict contains the resized image

    for k in image_data.keys():
        image_data_dict[k] = {**image_data_dict[k], **image_data[k]}  # merge the two dicts

    image_data_dict = {key.replace('.png', ''): value for key, value in image_data_dict.items()}

    for k, x in image_data_dict.items():
        image_data_dict[k]['white_mask'] = generate_image_binary_mask(x['image'], channel_first=False)

    image_data = np.array([x['image'] for k, x in image_data_dict.items()])
    image_means = np.mean(image_data, axis=(0, 1, 2))
    image_stds = np.std(image_data, axis=(0, 1, 2))
    for k, x in image_data_dict.items():
        image_data_dict[k]['image_z_normed'] = (x['image'] - image_means) / image_stds

    # make the image channel_first to be compatible with downstream training
    for k, x in image_data_dict.items():
        image_data_dict[k]['image_z_normed'] = image_data_dict[k]['image_z_normed'].transpose((2, 0, 1))

    trial_samples = []
    image_name_counts = defaultdict(int)

    # load gaze sequences
    if root_drive_path_cleaned is not None:
        def convert_img_name(input_img_name):
            """takes a image name [nw] d+ can turns it into (normal|wetAMD) d+"""
            # assert the input name matches the pattern [nw]\d+
            assert re.match(r'^[nw]\d+$', input_img_name), f"Input image name '{input_img_name}' does not match the pattern '[nw]\\d+'"
            return 'normal' + input_img_name[1:] if input_img_name[0] == 'n' else 'wetAMD' + input_img_name[1:]

        df_combined = pd.read_csv('all_data.csv') # load_all_fixations(root_drive_path_gaze, root_drive_path_cleaned) # load_all_karen_fixations(all_karen_tsv_fixation_path)
        df_filtered = df_combined[df_combined['image_name'].apply(lambda x: x is not None and x[0] in ['w', 'n'])]
        df_filtered.loc[:, 'grouped_image_name'] = df_filtered['image_name'].apply(lambda x: x.split('_')[0])
        df_filtered.loc[:, 'grouped_image_name'] = df_filtered['grouped_image_name'].apply(convert_img_name)
        df_filtered.loc[:, 'layer'] = df_filtered['image_name'].apply(lambda x: x.split('_')[1].strip('.png'))
        
        # get the presented media width and height
        stimulus_width = df_filtered['Presented Media width [px]'].unique()
        stimulus_height = df_filtered['Presented Media height [px]'].unique()
        assert len(stimulus_width) == 1
        assert len(stimulus_height) == 1
        stimulus_width = stimulus_width[0]
        stimulus_height = stimulus_height[0]

        image_height, image_width = image_data_dict[list(image_data_dict.keys())[0]]['sub_images'][0]['image'].shape[1:]
        n_patches_height, n_patches_width = int(image_height/patch_size[0]), int(image_width/patch_size[1])
        print('\n'*5)
        unique_images = df_filtered['grouped_image_name'].unique()
        print(f'Unique Images: {len(unique_images)}')
        for i, image_name in enumerate(unique_images):
            # Iterate over the layers/subimages for this image
            layers = df_filtered[df_filtered['grouped_image_name'] == image_name]['layer'].unique()
            print(f'Number of unique Layers for {image_name}: {len(layers)}')

            fixation_sequences = []
            aois = []

            # Ensure the layers are handled consistently (e.g., 1 to 5)
            max_layers = 5  # change as needed
            all_layers = range(1, max_layers + 1)  

            for layer in all_layers:
                if layer in layers:
                    # Get the fixation points for the current layer
                    fixations = df_filtered[
                        (df_filtered['grouped_image_name'] == image_name) & 
                        (df_filtered['layer'] == layer)
                    ][['Fixation point X [DACS px]', 'Fixation point Y [DACS px]']].values

                    # Filter out points outside the presented media (within the stimulus dimensions)
                    valid_fixations = fixations[
                        (fixations[:, 0] >= 0) & (fixations[:, 0] <= stimulus_width) &  # X within width
                        (fixations[:, 1] >= 0) & (fixations[:, 1] <= stimulus_height)  # Y within height
                    ]

                    # Append the valid fixation points sequence for this layer
                    fixation_sequences.append(valid_fixations)  # this is width, height

                    # Create AOI heatmap
                    aoi, xedges, yedges = np.histogram2d(
                        valid_fixations[:, 1],  # change this to height, width to match the image_data_dict's axis
                        valid_fixations[:, 0], 
                        bins=(n_patches_height, n_patches_width)
                    )
                    aoi = aoi / aoi.sum()  # Normalize
                else:
                    # Create a blank AOI array if the layer is missing
                    fixation_sequences.append(np.array([]))
                    aoi = np.zeros((n_patches_height, n_patches_width))

                aois.append(aoi)
            print(f'{i+1}. Image Data:')
            print(f'Image Name: {image_name}')
            print(f'AoIs: {len(aois)}')
            cnt = 0
            for aoi in aois:
                if len(aoi) > 0:
                    cnt += 1
                print(aoi.shape)
            print(f'AoIs that are complete: {cnt}\n\n')
            trial_samples.append({**{'name': image_name, 'fix_seq': fixation_sequences, 'aoi': aois}, **image_data_dict[image_name]})

    no_fixation_count = 0
    trial_samples_image_names = [x['name'] for x in trial_samples]
    for image_name, image_data in image_data_dict.items():
        if image_name not in trial_samples_image_names:
            trial_samples.append({**{'name': image_name, 'fix_seq': np.zeros((0, 2))}, **image_data})
            no_fixation_count += 1
    print(f"There are {no_fixation_count} images among {len(image_data_dict)} that doesn't have fixation data")

    print(f"Each image is used in on average:median {np.mean(list(image_name_counts.values()))}:{np.median(list(image_name_counts.values()))} trials")
    image_labels = np.array([v['label'] for v in image_data_dict.values()])
    unique_labels = np.unique(image_labels)

    # plt.bar(np.arange(len(unique_labels)), [np.sum(image_labels==l) for l in unique_labels])
    # plt.xlabel("Number of images")
    # plt.xticks(np.arange(len(unique_labels)), unique_labels)
    # plt.title("Number of images per label")
    # plt.savefig('OldData.png')

    # trial_labels = np.array([v['label'] for v in trial_samples])
    # plt.bar(np.arange(len(unique_labels)), [np.sum(trial_labels==l) for l in unique_labels])
    # plt.xlabel("Number of images")
    # plt.xticks(np.arange(len(unique_labels)), unique_labels)
    # plt.title("Number of trials per label")
    # plt.savefig('OldDataTrials.png')

    image_labels = np.array([v['label'] for v in image_data_dict.values()])
    
    return trial_samples, image_data_dict, image_labels, {'image_means': image_means, 'image_stds': image_stds,
                                                          'subimage_mean': subimage_loader.subimage_mean, 'subimage_std': subimage_loader.subimage_std,
                                                          'subimage_sizes': [x['image'].shape[1:] for x in trial_samples[0]['sub_images']]}

ans = get_bscan_test_train_val_folds(data_root, image_size=image_size, n_folds=folds, n_jobs=n_jobs,
                                                                                    cropped_image_data_path=cropped_image_data_path,
                                                                                    all_karen_tsv_fixation_path=all_karen_tsv_fixation_path,
                                                                                    root_drive_path_gaze=all_fixation_path_gaze,
                                                                                    root_drive_path_cleaned=all_fixation_path_cleaned_response,
                                                                                    patch_size=patch_size, gaussian_smear_sigma=gaussian_smear_sigma,
                                                                                    test_size=test_size, val_size=val_size)

# print(ans)

# use_saved_folds = '/data/rishabh/ExpertInformedDL_v3/folds'
# use_saved_folds = '/data/leo/temp/bscan/vit'

# print(f"Using saved folds from {use_saved_folds}")
# folds = pickle.load(open(os.path.join(use_saved_folds, 'folds.p'), 'rb'))
# test_dataset = pickle.load(open(os.path.join(use_saved_folds, 'test_dataset.p'), 'rb'))
# image_stats = pickle.load(open(os.path.join(use_saved_folds, 'image_stats.p'), 'rb'))
# test_dataset.compound_label_encoder = pickle.load(open(os.path.join(use_saved_folds, 'compound_label_encoder.p'), 'rb'))

# for fold_i, (train_trial_dataset, valid_dataset, train_unique_img_dataset) in enumerate(folds):

#     train_dataset = train_trial_dataset
#     # tt = train_trial_dataset[0]
#     # print(len(train_trial_dataset))

#     # for i, key in enumerate(tt):
#     #     print(f'Item {i}')
#     #     print(key)

#     #     print()
        
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_bscan)

#     # Assuming train_loader is already defined
#     for batch_idx, batch in enumerate(train_loader):
#         # Unpack the batch based on collate_fn's return structure
#         if len(batch) == 8:  # Includes sub_images
#             img, label, label_encoded, fixation_sequence, aoi_heatmap, image_resized, image_original, subimage_positions = batch
#         else:
#             img, label, label_encoded, fixation_sequence, aoi_heatmap, image_resized, image_original = batch

#         if aoi_heatmap is None:
#             continue
#         print(f"Batch {batch_idx} Details:")
#         print("Image shape:", img.shape if isinstance(img, torch.Tensor) else "Not a tensor")
#         print("Label shape:", label.shape)
#         print("Label Encoded shape:", label_encoded.shape)
#         print("Fixation Sequence Length:", len(fixation_sequence))
#         print("AOI Heatmap shape:", aoi_heatmap.shape if aoi_heatmap is not None else "None")
#         print("Resized Image shape:", image_resized.shape)
#         print("Original Image details:", type(image_original), len(image_original))
        
#         if len(batch) == 8:
#             print("Subimage Positions:", subimage_positions)

#         # Exit after first batch for brevity
#         break
#     break


# print('='*100)
# print()



# # for fold_i, (train_trial_dataset, valid_dataset, train_unique_img_dataset) in enumerate(folds):
# #     train_names = {x['name'] for x in train_trial_dataset.trial_samples}
# #     valid_names = {x['name'] for x in valid_dataset.trial_samples}

# #     train_unique_names = {x['name'] for x in train_unique_img_dataset.trial_samples}

# #     assert len(valid_names.intersection(train_names)) == 0
# #     assert len(valid_names.intersection(train_unique_names)) == 0

# #     tt = train_trial_dataset[0]
# #     # print(len(train_trial_dataset))

# #     for i, key in enumerate(tt):
# #         print(f'Item {i}')
# #         print(key)

# #         print()

# use_saved_folds = '/data/rishabh/ExpertInformedDL_v3/folds'
# # use_saved_folds = '/data/leo/temp/bscan/vit'

# print(f"Using saved folds from {use_saved_folds}")
# folds = pickle.load(open(os.path.join(use_saved_folds, 'folds.p'), 'rb'))
# test_dataset = pickle.load(open(os.path.join(use_saved_folds, 'test_dataset.p'), 'rb'))
# image_stats = pickle.load(open(os.path.join(use_saved_folds, 'image_stats.p'), 'rb'))
# test_dataset.compound_label_encoder = pickle.load(open(os.path.join(use_saved_folds, 'compound_label_encoder.p'), 'rb'))

# for fold_i, (train_trial_dataset, valid_dataset, train_unique_img_dataset) in enumerate(folds):

#     train_dataset = train_trial_dataset
#     # tt = train_trial_dataset[0]
#     # print(len(train_trial_dataset))

#     # for i, key in enumerate(tt):
#     #     print(f'Item {i}')
#     #     print(key)

#     #     print()
        
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_bscan)

#     # Assuming train_loader is already defined
#     for batch_idx, batch in enumerate(train_loader):
#         # Unpack the batch based on collate_fn's return structure
#         if len(batch) == 8:  # Includes sub_images
#             img, label, label_encoded, fixation_sequence, aoi_heatmap, image_resized, image_original, subimage_positions = batch
#         else:
#             img, label, label_encoded, fixation_sequence, aoi_heatmap, image_resized, image_original = batch

#         if aoi_heatmap is None:
#             continue
#         print(f"Batch {batch_idx} Details:")
#         print("Image shape:", img.shape if isinstance(img, torch.Tensor) else "Not a tensor")
#         print("Label shape:", label.shape)
#         print("Label Encoded shape:", label_encoded.shape)
#         print("Fixation Sequence Length:", len(fixation_sequence))
#         print("AOI Heatmap shape:", aoi_heatmap.shape if aoi_heatmap is not None else "None")
#         print("Resized Image shape:", image_resized.shape)
#         print("Original Image details:", type(image_original), len(image_original))
        
#         if len(batch) == 8:
#             print("Subimage Positions:", subimage_positions)

#         # Exit after first batch for brevity
#         break
#     break


# print('='*100)