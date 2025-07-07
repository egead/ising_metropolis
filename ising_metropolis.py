import numpy as np 
from numba import jit

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


@jit(nopython=True)
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
    random_col = np.random.choice(cols)
    print('Selected spin: ', (random_row, random_col))

    return [random_row,random_col]

@jit(nopython=True)
def calculate_total_energy(grid, J=1.0): 
    '''
    Calculates the total energy with coupling constant J. 
    J is the coupling constant. 
    J > 0 => ferromagnetic
    J < 0 => antiferromagnetic
    
    Args: 
        grid: np.array
        J: float
    Returns:
        total_energy:float
    '''
    L = grid.shape[0]
    total_energy=0.0
    for i in range(L):
        for j in range(L):
            down = (i+1)%L 
            right = (j+1)%L
            total_energy -= J*grid[i,j]*(grid[down,j]+grid[i,right])
    return total_energy 

@jit(nopython=True)
def calculate_energy_diff(grid, spin_coordinates, J):
    '''
    Calculates ∆E 

    ΔE=2J⋅s_i⋅∑_neighbours s_j
    
    Args: 
        spin_coordinates: list
        grid: np.array
        J: float
    
    Returns: 
        energy_diff: float
    '''
    L=grid.shape[0]
    i,j = spin_coordinates

    up = (i-1)%L
    down = (i+1)%L
    left = (j-1)%L
    right = (j+1)%L

    energy_diff = 2 *J* grid[i,j] * (grid[up,j] + grid[down,j]
                                + grid[i,left] + grid[i, right])
    
    print('\n Calculated energy diff for spin in coordinates ', spin_coordinates, ' is ', energy_diff)

    return energy_diff 

@jit(nopython=True)
def calculate_magnetization(grid):
    '''
    Calculates total magnetization
    '''
    return np.sum(grid)

@jit(nopython=True)
def calculate_specific_heat(energies, T, N):
    """
    Calculate specific heat from energy measurements
    Specific heat per spin: C_v = <(ΔE)²> / (k_B T² N)
    
    Args:
        energies: Array of energy measurements
        T: Temperature
        N: Total number of spins (L²)
    
    Returns:
        specific_heat: Specific heat per spin
    """
    mean_E = np.mean(energies)
    mean_E2 = np.mean(energies**2)
    variance_E = mean_E2 - mean_E**2
    C_v = variance_E / (T**2 * N)
    return C_v

@jit(nopython=True)
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
    x = np.random.rand()
    
    if energy_diff <= 0: 
        print('\n Energy difference is less than 0, decided TO FLIP')
        return 1
    else:
        if np.exp(-energy_diff/T)>x:
            print('\n exp(-energy_diff/T)>x, decided TO FLIP')
            return 1
        else: 
            print('\n exp(-energy_diff/T)<x, decided NOT TO FLIP')
            return 0



@jit(nopython=True)
def flip(grid, spin_coordinates):
    '''
    Flips the spin

    Args:
        grid: np.array
        spin_coordinates: list 
    Returns:
        grid: np.array
    '''

    grid[spin_coordinates[0], spin_coordinates[1]] = -grid[spin_coordinates[0], spin_coordinates[1]]

    return grid

@jit(nopython=True)
def metropolis_sweep(grid, T, J=1.0, n_steps=1000, sample_interval=100):
    """
    Metropolis simulation with observable tracking
    
    Args:
        grid: (np.array) Initial configuration
        T: (float) Temperature
        J: (float) Coupling constant  
        n_steps: (int) Total MC steps
        sample_interval: (int) How often to record observables
        
    Returns:
        grid: (np.array) Final configuration
        energies: (np.array) Array of energy measurements
        magnetizations: (np.array) Array of magnetization measurements
    """
    
    n_samples = n_steps // sample_interval
    energies = np.zeros(n_samples)
    magnetizations = np.zeros(n_samples)
    
    sample_count = 0
    
    for step in range(n_steps):
        print('Step: ', step)
        coordinates = select_random_spin(grid)
        delta_E = calculate_energy_diff(grid, coordinates, J)
        if decide_flipping(delta_E, T):
            flip(grid, coordinates)
        
        # Record observables periodically
        if step % sample_interval == 0 and sample_count < n_samples:
            energies[sample_count] = calculate_total_energy(grid, J)
            magnetizations[sample_count] = calculate_magnetization(grid)
            sample_count += 1
    
    return grid, energies, magnetizations


def autocorrelation_function(data, max_lag=None):
    """
    Calculate autocorrelation function
    
    Args:
        data: Time series data (e.g., energies or magnetizations)
        max_lag: Maximum lag to calculate (default: len(data)//4)
    
    Returns:
        autocorr: Autocorrelation function
        lags: Corresponding lag values
    """
    if max_lag is None:
        max_lag = len(data) // 4
    
    data_centered = data - np.mean(data)
    
    n = len(data_centered)
    autocorr = np.zeros(max_lag)
    variance = np.var(data_centered)
    
    for lag in range(max_lag):
        if lag < n:
            autocorr[lag] = np.mean(data_centered[:-lag if lag > 0 else None] * 
                                  data_centered[lag:]) / variance
    
    lags = np.arange(max_lag)
    return autocorr, lags

def integrated_autocorr_time(autocorr):
    """
    Calculate integrated autocorrelation time
    τ_int = 1 + 2 * Σ ρ(t) where ρ(t) > 0
    """
    tau_int = 1.0
    for i in range(1, len(autocorr)):
        if autocorr[i] <= 0:
            break
        tau_int += 2 * autocorr[i]
    
    return tau_int
