print("hello")

'''
we want to create a pickle file in the following format:
    Structure of the cropped_image_data: dict
            image_name: str: image names are the keys of the dict
                'image': np.array: the original image
                'sub_images': dict
                    'En-face_52.0micrometer_Slab_(Retina_View)': dict
                        'sub_image': np.array
                        'position': list of four two-int tuples
                    'Circumpapillary_RNFL':                             same as above
                    'RNFL_Thickness_(Retina_View)':                     same as above
                    'GCL_Thickness_(Retina_View)':                      same as above
                    'RNFL_Probability_and_VF_Test_points(Field_View)':  same as above
                    'GCL+_Probability_and_VF_Test_points':              same as above
                'label': str: 'G', 'S', 'G_Suspects', 'S_Suspects'
                # replace label with something else

'''
import os
import numpy as np
from PIL import Image
import pickle

# no_amd_path = '/data/kuang/David/ExpertInformedDL_v3/non-AMD'
# amd_path = '/data/kuang/David/ExpertInformedDL_v3/AMD'

img_folder = '/data/rishabh/ExpertInformedDL_v3/amd_images'

image_dict = {}

for file in os.listdir(img_folder):
    if file.endswith('.png'):
        img_path = os.path.join(img_folder, file)
        with Image.open(img_path) as img:
            print(img_path)
            image_dict[file] = {}
            image_dict[file]['original_image'] = np.array(img)
            image_dict[file]['label'] = 'N' if file[0] == 'n' else 'A'

with open('/data/rishabh/ExpertInformedDL_v3/bscan_imgs.p','wb') as file:
    pickle.dump(image_dict, file)





    

