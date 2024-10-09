# Expert-attention guided deep learning for medical images

## Get Started

Pip install the PYPI distro:

```bash
pip install expert-informed-dl
```

### Here's an example of how to use the trained model for inference (with subimages)

Check out eidl/examples/subimage_example.py for a simple example of how to use the trained model for inference on subimages.

```python
from eidl.utils.model_utils import get_subimage_model

subimage_handler = get_subimage_model()
subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', is_plot_results=True, discard_ratio=0.1)

```

### If you want to use the rollouts/gradcams in a user interface, you may consider precomputing them, as it can be slow to compute them on the fly.

```python
from eidl.utils.model_utils import get_subimage_model

subimage_handler = get_subimage_model(precompute='vit')

# or

subimage_handler = get_subimage_model(precompute='resnet')

# or

subimage_handler = get_subimage_model(precompute=['vit', 'resnet'])

```


### If you don't want to use subimages:

Check out eidl/examples/example.py for a simple example of how to use the trained model for inference.

When forwarding image through the network, use the argument `collapse_attention_matrix=True` to get the attention matrix
to get the attention matrix averaged across all heads and keys for each query token. 

```python
y_pred, attention_matrix = model(image_data, collapse_attention_matrix=False)

```


# Train model locally

Install `requirements.txt` by running the command below:

```bash
pip install -r requirements.txt
```

Torch is not included in requirements. 
Please download Pytorch matching with a CUDA version matching your GPU from [here](https://pytorch.org/get-started/locally/).


## Train on the OCT dataset

The main entry point for training on the OCT dataset is main_oct_v3.py. This script runs a 
grid search over the hyperparameters and saves the best model to disk.

This script needs to load data from oct_v2. Download the data from [here](https://www.dropbox.com/scl/fo/ici56fa8w6knjyxaer72f/AM_HG1ERhiDpLfda07U8Xfg?rlkey=vbtd1gz5iixh5uk8qfaog9jv9&dl=0).
Change the data path and accordingly in the script.

**Use saved folds** when running the script for the first time. You should set `use_saved_folds=None` in the script. This way
it will save the folds to disk. When you run the script again, you can set `use_saved_folds=<path to the saves>` to use the saved folds.
The results will be saved to `results_dir`. It is recommended you change the name of the results directory when one run is complete 
to avoid overwriting the results.

## Visualize the running results

To visualize the training results, including the train history and the attention rollout or gradcam. Run the script
`main_oct_viz.py`. You will need to change the `results_dir` to match the one you used in `main_oct_v3.py`.
You can also set a `figure_dir` to save the figures to disk.

```bash

All the training parameters including the grid search values are defined before the main clause. 
Feel free to change them as needed.


# API Reference

model.forward(collapse_attention_matrix)

For example, if you have 32 * 32 patches,
the attention matrix will be of size (32 * 32 + 1) 1025. Plus one for the classificaiton token.
If you set `collapse_attention_matrix=False`, the attention matrix will be
uncollapsed. The resulting attention matrix will be of shape (n_batch, n_heads, n_queries, n_keys). For example, if you have 32 * 32 patches,
one image and one head, the attention matrix will be of shape (1, 1, 1025, 1025).


## Troubleshoot

If get model functions raises the following error:

```bash
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```

You will need to install the correct version of Pytorch matching with a CUDA version matching your GPU from [here](https://pytorch.org/get-started/locally/).
This is because all the models are trained on GPU.

# BScans

The five layers of the BScan is place horizontally with layer 1 at the left, and layer 5 at the right. This forms
a wide image.
Use the script `bscan_v1` to train the model on BScans. The script is similar to `main_oct_v3.py` but with the necessary changes to train on BScans.