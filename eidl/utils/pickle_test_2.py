import pickle

# Replace 'your_file.pkl' with the path to your pickle file
with open('/data/kuang/David/ExpertInformedDL_v3/bscan_v2.p', 'rb') as f:
    data = pickle.load(f)

# Find the length of the dictionary
length = len(data)

# Print the keys of the dictionary
keys = data.keys()

print(f"Length of dictionary: {length}")
print("Keys in the dictionary:")
for key in keys:
    print(key)
