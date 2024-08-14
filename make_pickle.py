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

no_amd_path = '/data/kuang/David/ExpertInformedDL_v3/non-AMD'
amd_path = '/data/kuang/David/ExpertInformedDL_v3/AMD'

image_dict = {}

for file in os.listdir(no_amd_path):
    if file.endswith('.png'):
        img_path = os.path.join(no_amd_path, file)
        with Image.open(img_path) as img:
            print(img_path)
            image_dict[file] = {}
            sub_dict = image_dict[file]
            sub_dict['original_image'] = np.array(img)
            sub_dict['sub_images'] = {}
            sub_dict['label'] = 'A'
            sub_dict2 = sub_dict['sub_images']
            list = ['bscan1', 'bscan2', 'bscan3', 'bscan4', 'bscan5']

            for i,bscan in enumerate(list):
                sub_dict2[bscan] = {}
                sub_dict3 = sub_dict2[bscan]
                left = i * 1055
                right = left + 1055
                top = 0
                bottom = 703
                subimage = img.crop((left, top, right, bottom))
                sub_dict3['sub_image'] = np.array(subimage)
                position = [(left, top), (right, top), (right, bottom), (left, bottom)]
                sub_dict3['position'] = position

for file in os.listdir(amd_path):
    if file.endswith('.png'):
        img_path = os.path.join(amd_path, file)
        with Image.open(img_path) as img:
            print(img_path)

            image_dict[file] = {}
            sub_dict = image_dict[file]
            try:
                sub_dict['original_image'] = np.array(img)
            except:
                print(img)
            sub_dict['sub_images'] = {}
            sub_dict['label'] = 'N'
            sub_dict2 = sub_dict['sub_images']
            list = ['bscan1', 'bscan2', 'bscan3', 'bscan4', 'bscan5']
            for i,bscan in enumerate(list):
                sub_dict2[bscan] = {}
                sub_dict3 = sub_dict2[bscan]
                left = i * 1055
                right = left + 1055
                top = 0
                bottom = 703
                subimage = img.crop((left, top, right, bottom))
                sub_dict3['sub_image'] = np.array(subimage)
                position = [(left, top), (right, top), (right, bottom), (left, bottom)]
                sub_dict3['position'] = position

with open('/data/kuang/David/ExpertInformedDL_v3/bscan_v2.p','wb') as file:
    pickle.dump(image_dict, file)





    

