import numpy as np 

def create_random_grid(grid_shape):
    '''
    Creates a random grid (np.array) of size grid_shape.
    Each entry can be either -1 or 1

    Args: 
        grid_shape: (n_rows, n_columns)

    Returns: 
        np.array
    '''
    return np.random.choice([-1, 1], size=grid_shape)

def create_uniform_grid(grid_shape, direction: int = 1):
    '''
    Creates a uniform grid (np.array) of size grid_shape. 
    All entries take "direction". Direction must be either -1 or 1

    Args: 
        grid_shape: (n_rows, n_columns)
        direction: int

    Returns:
        np.array
    '''
    if direction == 1: 
        return np.ones(size=grid_shape, dtype=int)
    elif direction == -1: 
        return np.full(size=grid_shape, fill_value = -1, dtype=int)
    else: 
        raise ValueError("Direction can be only -1 or 1")


