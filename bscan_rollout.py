import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from eidl.utils.model_utils import parse_model_parameter, get_best_model, parse_training_results

from eidl.utils.model_utils import get_trained_model, load_image_preprocess
from eidl.viz.vit_rollout import VITAttentionRollout

# replace the image path to yours
# set image size

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


results_dir = 'temp/results-bscan-07_17_2024_00_50_17'
image_path = 'AMD/169_1_OD.tiff_3channel.png'

results_dict, model_config_strings = parse_training_results(results_dir)
image_stats = pickle.load(open(os.path.join(results_dir, 'image_stats.p'), 'rb'))

models = {parse_model_parameter(x, 'model') for x in model_config_strings}
models = list(reversed(list(models)))
best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict)
best_model.to(device)
best_model.eval()


alphas = {parse_model_parameter(x, 'alpha') for x in model_config_strings}
alphas = list(alphas)
alphas.sort()

model = best_model
model.to(device)


compound_label_encoder = pickle.load(open('/data/kuang/David/ExpertInformedDL_v3/temp/results-bscan-07_17_2024_00_50_17/compound_label_encoder.p','rb'))

image_stats = pickle.load(open('/data/kuang/David/ExpertInformedDL_v3/temp/results-bscan-07_17_2024_00_50_17/image_stats.p','rb'))



image_std = image_stats['subimage_std']

image_mean = image_stats['subimage_mean']
image_size = 5275,703



#mage_mean, image_std, image_size = get_trained_model(device, model_param='num-patch-32_image-size-1024-512')

image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std)
print(image.size)
# get the prediction
image_tensor = torch.Tensor(image_normalized).unsqueeze(0).to(device)
y_pred, attention_matrix = model(image_tensor, collapse_attention_matrix=False)
predicted_label = np.array([torch.argmax(y_pred).item()])

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
plt.savefig("akdsfmka.png")