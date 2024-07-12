import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from eidl.utils.model_utils import get_trained_model, load_image_preprocess, get_model
from eidl.viz.vit_rollout import VITAttentionRollout
from eidl.viz.viz_oct_results import viz_oct_results
import os
import pickle

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from PIL import Image

from eidl.Models.ExtensionModel import ExtensionModelSubimage, get_gradcam
from eidl.datasets.OCTDataset import OCTDatasetV3
from eidl.utils.image_utils import process_aoi, process_grad_cam
from eidl.utils.iter_utils import collate_fn
from eidl.utils.model_utils import parse_model_parameter, get_best_model, parse_training_results
from eidl.utils.torch_utils import any_image_to_tensor
from eidl.utils.training_utils import run_one_epoch, run_one_epoch_oct
from eidl.viz.vit_rollout import VITAttentionRollout

from eidl.viz.viz_utils import plt2arr, plot_train_history, plot_subimage_rolls, plot_image_attention, \
    register_cmap_with_alpha, recover_subimage

# replace the image path to yours

viz_oct_results('/data/kuang/David/ExpertInformedDL_v3/temp/results-07_11_2024_05_31_47', 1)

#viz_oct_results('/data/kuang/David/ExpertInformedDL_v3/temp/results-07_08_2024_11_17_34', 1)


######
'''
import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM


from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

path = ''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


    #     python vit_gradcam.py --image-path <path_to_image>
    # Example usage of using cam-methods on a VIT network.

    

methods = \
    {"gradcam": GradCAM}
n_jobs=1
acc_min=.3
acc_max=1
viz_val_acc=False
plot_format='individual'
num_plot=14,
rollout_transparency=0.75
figure_dir='/data/kuang/David/ExpertInformedDL_v3/temp/results-07_08_2024_11_17_34/figures'
results_dir = '/data/kuang/David/ExpertInformedDL_v3/temp/results-07_10_2024_01_42_56'
image_stats = pickle.load(open(os.path.join(results_dir, 'image_stats.p'), 'rb'))
print(f'image_stats {image_stats}')
# load the test dataset ############################################################################################
test_dataset = pickle.load(open(os.path.join(results_dir, 'test_dataset.p'), 'rb'))
folds = pickle.load(open(os.path.join(results_dir, 'folds.p'), 'rb'))

results_dict, model_config_strings = parse_training_results(results_dir)

# np.random.choice([x['name'] for x in test_dataset.trial_samples if x['label'] == 'G'], size=16, replace=False)
# np.random.choice([x['name'] for x in test_dataset.trial_samples if x['label'] == 'S'], size=16, replace=False)

# results_df.to_csv(os.path.join(results_dir, "summary.csv"))

# run the best model on the test set
models = {parse_model_parameter(x, 'model') for x in model_config_strings}
models = list(reversed(list(models)))
best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict)
best_model.eval()

model = torch.hub.load('facebookresearch/deit:main',
                        'deit_tiny_patch16_224', pretrained=True)
model.eval()

target_layers = [model.blocks[-1].norm1]

if args.method not in methods:
    raise Exception(f"Method {args.method} not implemented")

if args.method == "ablationcam":
    cam = methods[args.method](model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda,
                                reshape_transform=reshape_transform,
                                ablation_layer=AblationLayerVit())
else:
    cam = methods[args.method](model=model,
                                target_layers=target_layers,
                                use_cuda=args.use_cuda,
                                reshape_transform=reshape_transform)

rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested category.
targets = None

# AblationCAM and ScoreCAM have batched implementations.
# You can override the internal batch size for faster computation.
cam.batch_size = 32

grayscale_cam = cam(input_tensor=input_tensor,
                    targets=targets,
                    eigen_smooth=args.eigen_smooth,
                    aug_smooth=args.aug_smooth)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite(f'{args.method}_cam.jpg', cam_image)




################

#image_path = r'D:\Dropbox\Dropbox\ExpertViT\Datasets\OCTData\oct_v2\reports_cleaned\G_Suspects\RLS_074_OD_TC.jpg'
'''

'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#get model
image_size = 
model = get_model()
image_mean = 
image_std = 
image_size = 
compound_label_encoder = 

#model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device, model_param='num-patch-32_image-size-1024-512')

image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std)

# get the prediction
image_tensor = torch.Tensor(image_normalized).unsqueeze(0).to(device)
y_pred, attention_matrix = model(image_tensor, collapse_attention_matrix=False)
predicted_label = np.array([torch.argmax(y_pred).item()])
decoded_label = compound_label_encoder.decode(predicted_label)

print(f'Predicted label: {decoded_label}')

# plot the attention rollout
vit_rollout = VITAttentionRollout(model, device=device, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.5)
rollout = vit_rollout(depth=model.depth, input_tensor=image_tensor)  # rollout on the last layer

rollout_resized = cv2.resize(rollout, dsize=image_size, interpolation=cv2.INTER_LINEAR)
rollout_heatmap = cv2.applyColorMap(cv2.cvtColor((rollout_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET)
rollout_heatmap = cv2.cvtColor(rollout_heatmap, cv2.COLOR_BGR2RGB)
alpha = 0.2
output_image = cv2.addWeighted(image.astype(np.uint8), alpha, rollout_heatmap, 1 - alpha, 0)


fig = plt.figure(figsize=(15, 30), constrained_layout=True)
axes = fig.subplots(3, 1)
axes[0].imshow(image.astype(np.uint8))  # plot the original image
axes[0].axis('off')
axes[0].set_title(f'Original image')

axes[1].imshow(rollout_heatmap)  # plot the attention rollout
axes[1].axis('off')
axes[1].set_title(f'Attention rollout')

axes[2].imshow(output_image)  # plot the attention rollout
axes[2].axis('off')
axes[2].set_title(f'Overlayed attention rollout')
plt.show()
'''''