import functools

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
