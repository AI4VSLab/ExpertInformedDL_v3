import pickle

with open('/home/kavin/ExpertInformedDL_v3/bscan_imgs.p', 'rb') as f:
    data = pickle.load(f)

print(f"Keys: {list(data.keys())[:5]}")  # sample keys
print(f"Sample inner keys: {list(data[next(iter(data))].keys())}")  # e.g., 'original_image'
