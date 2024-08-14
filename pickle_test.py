import pickle
import numpy as np

path = '/data/kuang/David/ExpertInformedDL_v3/oct_reports_info_repaired.p'
with open(path, 'rb') as file:
    data = pickle.load(file)

positions = []
values = data.values()
for key,value in data.items():
    print(key)
    print(type(value))
    print(value['sub_images'])

for position in positions:
    print(position)




