import functools
import zipfile
import pandas as pd

# Shameless stolen from the comments of 
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
