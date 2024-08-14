import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle, os

from eidl.utils.model_utils import get_trained_model, load_image_preprocess, get_model, get_best_model, parse_training_results, parse_model_parameter
from eidl.viz.vit_rollout import VITAttentionRollout



#cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = torch.load('/data/kuang/David/ExpertInformedDL_v3/temp/results-07_10_2024_01_42_56/best_model-vit_small_patch32_224_in21k_subimage_alpha-0.75_dist-cross-entropy_lr-0.0001_depth-12_fold_0.pt')

results_dir = '/data/kuang/David/ExpertInformedDL_v3/temp/results-07_10_2024_01_42_56'

image_stats = pickle.load(open(os.path.join(results_dir, 'image_stats.p'), 'rb'))
# load the test dataset ############################################################################################
test_dataset = pickle.load(open(os.path.join(results_dir, 'test_dataset.p'), 'rb'))
folds = pickle.load(open(os.path.join(results_dir, 'folds.p'), 'rb'))

results_dict, model_config_strings = parse_training_results(results_dir)

print(results_dict)

print(model_config_strings)


models = {parse_model_parameter(x, 'model') for x in model_config_strings}
models = list(reversed(list(models)))

best_model, best_model_results, best_model_config_string = get_best_model(models, results_dict)

best_model.eval()

print(image_stats)


'''
def viz_vit_rollout(best_model, best_model_config_string, device, plot_format, num_plot, test_loader, has_subimage, figure_dir,
                    cmap_name, rollout_transparency, roll_image_folder, image_stats, *args, **kwargs):
    test_loader.dataset.create_aoi(best_model.get_grid_size())

    if hasattr(best_model, 'patch_height'):
        patch_size = best_model.patch_height, best_model.patch_width
    else:
        patch_size = best_model.vision_transformer.patch_embed.patch_size[0], best_model.vision_transformer.patch_embed.patch_size[1]

    model_depth = best_model.depth

    _rollout_info = []
    
    with torch.no_grad():

        # use gradcam if model is not a ViT
        vit_rollout = VITAttentionRollout(best_model, device=device, attention_layer_name='attn_drop', head_fusion="max", discard_ratio=0.0)
        sample_count = 0
        print(plot_format)
        if plot_format == 'grid':
            fig, axs = plt.subplots(model_depth + 2, num_plot, figsize=(2 * num_plot, 2 * (model_depth + 2)))
            plt.setp(axs, xticks=[], yticks=[])
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
            fig.tight_layout()

        for batch in test_loader:
            print(f'Processing sample {sample_count}/{len(test_loader)} in test set')
            image, image_resized, aoi_heatmap, subimages, subimage_masks, subimage_positions, image_original, image_original_size, label_encoded = process_batch(batch, has_subimage, device)

            # roll_depths = vit_rollout(depth=np.arange(best_model.depth), in_data=image)
            roll_depths = vit_rollout(depth=best_model.depth-1, in_data=image)

            if plot_format == 'individual':
                plot_original_image(image_original, image_original_size, aoi_heatmap, sample_count, figure_dir,
                                    has_subimage, best_model.get_grid_size(),
                                    subimages, subimage_masks, subimage_positions, patch_size, cmap_name,
                                    rollout_transparency)

                if type(roll_depths) is not list:
                    roll_depths = [roll_depths]
                for i, roll in enumerate(roll_depths):
                    rollout_image, subimage_roll = process_aoi(roll, image_original_size, has_subimage,
                                               grid_size=best_model.get_grid_size(),
                                               subimage_masks=subimage_masks, subimages=subimages,
                                               subimage_positions=subimage_positions, patch_size=patch_size, *args, **kwargs)

                    plot_image_attention(image_original, rollout_image, None, cmap_name='plasma',
                                         notes=f'#{sample_count} model {best_model_config_string}, roll depth {i}', save_dir=roll_image_folder)
                    plot_subimage_rolls(subimage_roll, subimages, subimage_positions, image_stats['subimage_std'], image_stats['subimage_mean'], cmap_name='plasma',
                                        notes=f"#{sample_count} model {best_model_config_string}, roll depth {i}", overlay_alpha=rollout_transparency, save_dir=roll_image_folder)
                _rollout_info.append([subimage_roll, subimages, subimage_positions])

                    # fig.savefig(f'figures/valImageIndex-{sample_count}_model-{model}_rollDepth-{i}.png')
                    # fig_list.append(plt2arr(fig))
                # imageio.mimsave(f'gifs/model-{model}_valImageIndex-{sample_count}.gif', fig_list, fps=2)  # TODO expose save dir
            elif plot_format == 'grid' and sample_count < num_plot:
                    axis_original_image, axis_aoi_heatmap, axes_roll = axs[0, sample_count], axs[1, sample_count], axs[2:, sample_count]
                    axis_original_image.imshow(image_original)  # plot the original image
                    axis_original_image.axis('off')
                    # axis_original_image.title(f'#{sample_count}, original image')

                    # plot the original image with expert AOI heatmap
                    axis_aoi_heatmap.imshow(image_original)  # plot the original image
                    _aoi_heatmap = cv2.resize(aoi_heatmap.numpy(), dsize=image.shape[1:], interpolation=cv2.INTER_LANCZOS4)
                    axis_aoi_heatmap.imshow(_aoi_heatmap.T, cmap=cmap_name, alpha=rollout_transparency)
                    axis_aoi_heatmap.axis('off')
                    # axis_aoi_heatmap.title(f'#{sample_count}, expert AOI')

                    for i, roll in enumerate(roll_depths):
                        rollout_image = cv2.resize(roll, dsize=image.shape[1:], interpolation=cv2.INTER_LANCZOS4)
                        axes_roll[i].imshow(np.moveaxis(image_resized, 0, 2))  # plot the original image
                        axes_roll[i].imshow(rollout_image.T, cmap=cmap_name, alpha=rollout_transparency)
                        axes_roll[i].axis('off')
                        # axes_roll[i].title(f'#{sample_count}, model {model}, , roll depth {i}')
            sample_count += 1
        viz_subimage_attention_grid(*zip(*_rollout_info), image_stats['subimage_std'], image_stats['subimage_mean'],
                                    roll_image_folder)
    print("hello")
    if plot_format == 'grid':
        plt.show()
    plt.savefig('rollout.png')  


'''