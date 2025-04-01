print("inside make_pickle.py file")

import os
import numpy as np
from PIL import Image
import pickle

# no_amd_path = '/data/kuang/David/ExpertInformedDL_v3/non-AMD'
# amd_path = '/data/kuang/David/ExpertInformedDL_v3/AMD'

img_folder = '/media/16TB_Storage/CenteredData/AMD_Dataset/images_subset_rishabh'

image_dict = {}

for file in os.listdir(img_folder):
    if file.endswith('.png'):
        img_path = os.path.join(img_folder, file)
        with Image.open(img_path) as img:
            print(img_path)
            image_dict[file] = {}
            image_dict[file]['original_image'] = np.array(img)
            image_dict[file]['label'] = 'N' if file[0] == 'n' else 'A'

with open('/home/kavin/ExpertInformedDL_v3/bscan_imgs.p','wb') as file:
    pickle.dump(image_dict, file)





    

