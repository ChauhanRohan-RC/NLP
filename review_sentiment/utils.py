import os
from collections.abc import Iterable

import numpy as np
import pandas.api.types
from pandas import DataFrame


def set_tensorflow_logging_level(level: int):  # level [0, 3] from most to least verbose
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


def lerp(start, end, factor):
    return start + ((end - start) * factor)


def get_lerp_factor(start, end, value):
    return (value - start) / (end - start)


def map_range(value, start, end, new_start, new_end):
    return lerp(new_start, new_end, get_lerp_factor(start, end, value))


def get_cluster_cohesion_factor(silhouette_score):
    """
    :param silhouette_score: silhouette_score in range [-1, 1] (high score means clusters are well separated from
    each other -> high cohesion)

    :return: cohesion factor in range (0, 1)
    """
    return get_lerp_factor(-1, 1, silhouette_score)


def digitize(arr: Iterable, start=0, step=1, return_mapping=False):
    un = np.unique(arr, return_counts=False)
    map_dic = {}
    for i, v in enumerate(un):
        map_dic[v] = start + (i * step)

    vec = np.vectorize(lambda x: map_dic[x])
    res = vec(arr)

    if return_mapping:
        return res, map_dic
    return res


def digitize_data_frame(df: DataFrame, exclude_columns: Iterable = None, exclude_dtypes: Iterable = None,
                        return_mappings: bool = False):
    cols = df.columns
    if cols.empty:
        return

    if exclude_columns:
        try:
            itr = iter(exclude_columns)
        except TypeError:
            # cols = tuple(filter(lambda x: x != exclude_columns, cols))
            pass
        else:
            _l = tuple(itr)
            cols = tuple(filter(lambda x: x not in _l, cols))

    exc_types = None
    if exclude_dtypes:
        try:
            itr = iter(exclude_dtypes)
        except TypeError:
            # exc_types = (exclude_dtypes, )
            pass
        else:
            exc_types = tuple(itr)

    map_dic = {} if return_mappings else None

    for c in cols:
        dtype = df[c].dtype
        if pandas.api.types.is_numeric_dtype(dtype) or (exc_types and dtype in exc_types):
            continue

        mapped = digitize(df[c], return_mapping=return_mappings)
        if return_mappings:
            df[c] = mapped[0]
            map_dic[c] = mapped[1]
        else:
            df[c] = mapped

    if return_mappings:
        return map_dic


def inverse_dict(dict_: dict):
    return dict(((v, k) for k, v in dict_.items()))
