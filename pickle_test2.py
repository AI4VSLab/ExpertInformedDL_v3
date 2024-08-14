import pickle

# Specify the path to your pickle file
pickle_file_path = '/data/kuang/David/ExpertInformedDL_v3/oct_reports_info_repaired.p'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)


def get_all_key_paths(d, path=None):
    """
    Recursively get all key paths in a nested dictionary.
    
    :param d: The dictionary to search through.
    :param path: The current path being traversed (used for recursion).
    :return: A list of all key paths.
    """
    if path is None:
        path = []

    paths = []

    if isinstance(d, dict):
        for key, value in d.items():
            new_path = path + [key]
            if isinstance(value, dict):
                paths.extend(get_all_key_paths(value, new_path))
            else:
                paths.append(new_path)

    return paths


# results = get_all_key_paths(data)
# for result in results:
#     print(f"Path: {' -> '.join(result)}")

'''
first_key = next(iter(data))
first_value = data[first_key]

subimages_value = first_value['subimages']

circumpapillary_value = subimages_value['Circumpapillary_RNFL']

print(circumpapillary_value)

'''