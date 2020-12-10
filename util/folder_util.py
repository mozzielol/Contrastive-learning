import os
import pickle
from pathlib import Path


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parser_bool(flag):
    if flag == 'True' or flag is True:
        return True
    elif flag == 'False' or flag is False:
        return False
    return None


def _iter_directory(directory, file_type='csv'):
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".{}".format(file_type)) and not filename.startswith("total"):
            file_list.append(filename)
        else:
            continue
    return file_list


