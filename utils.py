# * *************************************************************************
#   Programmer[s]: Leandro Fernandes
#   email: leandroohf@gmail.com
#   Program: utils
#   Commentary: My utils to help with processing data
#   Reference: https://www.thoughtworks.com/insights/blog/coding-habits-data-scientists
#   Date: February 6, 2020
#
#   The author believes that share code and knowledge is awesome.
#   Feel free to share and modify this piece of code. But don't be
#   impolite and remember to cite the author and give him his credits.
# * *************************************************************************

import functools
import zipfile
import pandas as pd
import numpy as np

# Shameless inspired from the comments of
# https://www.thoughtworks.com/insights/blog/coding-habits-data-scientists
def compose(*functions):
    """Ex: prepare_data = compose(functools.partial(encode_column, col_name='item'),
                       add_categorical_column,
                       convert_to_minutes
                      )
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def function_with_args(function, *args, **kargs):

    return functools.partial(function, *args, **kargs)

def load_zipped_data(data_file_name: str,csv_basename: str ):

    zf = zipfile.ZipFile(data_file_name)
    data = pd.read_csv(zf.open(csv_basename), index_col=False, encoding='latin1')

    return data

def print_7_number(data):

    # calculate quartiles
    quartiles = np.percentile(data, [25, 50, 75])
    # calculate min/max
    data_min, data_max = data.min(), data.max()

    _five_numbers = {'min': data_min,
                     'q1': quartiles[0],
                     'mean': np.mean(data),
                     'median': quartiles[1],
                     'q3': quartiles[2],
                     'max': data_max,
                     'std': np.std(data)
    }


    return _five_numbers
