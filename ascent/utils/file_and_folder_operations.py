import json
import os
import pickle  # nosec B403
from pathlib import Path
from typing import List
import glob

def load_pickle(file: str, mode: str = "rb"):  # nosec B301
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def save_pickle(obj, file: str, mode: str = "wb") -> None:  # nosec B301
    with open(file, mode) as f:
        pickle.dump(obj, f)


def load_json(file: str):
    with open(file) as f:
        a = json.load(f)
    return a


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def subdirs(
    folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> List[str]:
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfiles(
    folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> List[str]:
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    # recursive in structure of folder
    # res = [l(folder, os.path.relpath(i, folder))
    #        for i in glob.iglob(folder + '**/**', recursive=True)
    #        if os.path.isfile(i)
    #        and (prefix is None or i.startswith(prefix))
    #        and (suffix is None or i.endswith(suffix))
    # ]
    # res = list(set(res))
    if sort:
        res.sort()
    return res


def remove_suffixes(filename: Path) -> Path:
    """Removes all the suffixes from `filename`, unlike `stem` which only removes the last suffix.

    Args:
        filename: Filename from which to remove all extensions.

    Returns:
        Filename without its extensions.
    """
    return Path(str(filename).removesuffix("".join(filename.suffixes)))
