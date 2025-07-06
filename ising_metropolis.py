import numpy as np 

# I have the idea that maybe making this object oriented would be better
# However, I have to look if it would still be compatible with NUMBA

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

def create_uniform_grid(grid_shape, spin: int = 1):
    '''
    Creates a uniform grid (np.array) of size grid_shape. 
    All entries take "direction". Direction must be either -1 or 1

    Args: 
        grid_shape: (n_rows, n_columns)
        direction: int

    Returns:
        np.array
    '''
    if spin == 1: 
        return np.ones(shape=grid_shape, dtype=int)
    elif spin == -1: 
        return np.full(shape=grid_shape, fill_value = -1, dtype=int)
    else: 
        raise ValueError("Direction can be only -1 or 1")

def select_random_spin(grid):
    '''
    Selects a random spin from the given grid

    Args:
        grid: np.array
    
    Returns: 
        list: [ random_spin_coordinates]
    '''
    rows, cols = grid.shape

    random_row = np.random.choice(rows)
    random_col = np.random.shocide(cols)

    random_element =  grid[random_row, random_col]
    random_spin_coordinates = (random_row, random_col)
    return [random_element, random_spin_coordinates]

def calculate_energy_diff(spin_coordinates):
    '''
    Calculates ∆E 
    
    Args: 
        spin_coordinates
    
    Returns: 
        energy_diff: float
    '''

    return energy_diff 

def decide_flipping(energy_diff, T):
    '''
    Decides if the energy difference is sufficient for flipping the spin. 
    If ∆E ≤ 0, the change is favourable and the spin is flipped. 
    If ∆E > 0, then the spin is flipped only if exp(−∆E/kB T ) > x, 
    where x is a random number on the interval [0, 1] 

    Args: 
        energy_diff: float
        T: float 
    Returns: 
        int (0 or 1, 0 for no flip, 1 for flip.

    '''
    kB = 1 # Boltzmann constant 
    x = np.random.rand([0,1])
    
    if energy_diff <= 0: 
        return 1 
    elif energy_diff > 0: 
        if np.exp(-energy_diff/(kB*T)) > x:
            return 1
        elif np.exp(-energy_diff/(kB*T)) < x:
            return 0
        else:
            raise ValueError
    else:
        raise ValueError

def flip(grid, spin_coordinates, spin_value):
    '''
    Flips the spin
    '''

    return None

