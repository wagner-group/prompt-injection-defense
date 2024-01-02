import os

import dill


def wrapper(function_call, path, filename, force=False):
    if os.path.exists(os.path.join(path, filename)) and not force:
        with open(os.path.join(path, filename), "rb") as f:
            rtn = dill.load(f)
    else:
        rtn = function_call()
        with open(os.path.join(path, filename), "wb") as f:
            dill.dump(rtn, f)
    return rtn


def setup_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
