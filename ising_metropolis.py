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
    #print('Selected spin: ', (random_row, random_col))

    return [random_row,random_col]

@jit(nopython=True)
def calculate_total_energy(grid, J=1.0, lattice_type=0): 
    '''
    Calculates the total energy with coupling constant J. 
    J is the coupling constant. 
    J > 0 => ferromagnetic
    J < 0 => antiferromagnetic.

    Lattice type is 0 for square (default) and 1 for triangular

    total_energy is divided to two since each bond is counted twice.
    
    Args: 
        grid: np.array
        J: float
        lattice_type: int
    Returns:
        total_energy:float
    '''
    L = grid.shape[0]
    total_energy = 0.0
    
    for i in range(L):
        for j in range(L):
            
            if lattice_type == 0:  
                neighbors = [
                    ((i-1) % L, j), ((i+1) % L, j),
                    (i, (j-1) % L), (i, (j+1) % L)
                ]
            elif lattice_type == 1:
                up = ((i-1) % L, j)
                down = ((i+1) % L, j)
                left = (i, (j-1) % L)
                right = (i, (j+1) % L)
                
                if i % 2 == 1:  # Odd row
                    upper_left = ((i-1) % L, (j-1) % L)
                    lower_left = ((i+1) % L, (j-1) % L)
                    neighbors = [up, down, left, right, upper_left, lower_left]
                else:  # Even row
                    upper_right = ((i-1) % L, (j+1) % L)
                    lower_right = ((i+1) % L, (j+1) % L)
                    neighbors = [up, down, left, right, upper_right, lower_right]
            else: 
                raise ValueError("Lattice Type can only be 'square' (0) or 'triangular' (1)")
            
            # Sum energy contributions
            for ni, nj in neighbors:
                total_energy -= J * grid[i, j] * grid[ni, nj]
    
    return total_energy / 2.0

@jit(nopython=True)
def calculate_energy_diff(grid, spin_coordinates, J, lattice_type = 0 ):
    '''
    Calculates ∆E for square and triangular lattices.

    ΔE=2J⋅s_i⋅∑_neighbours s_j
    
    Args: 
        spin_coordinates: list
        grid: np.array
        J: float
        lattice_type: int (can be square (0) or triangular (1))
    
    Returns: 
        energy_diff: float
    '''
    L = grid.shape[0]
    i, j = spin_coordinates
    
    if lattice_type == 0: 
        neighbors = [
            ((i-1) % L, j),      # up
            ((i+1) % L, j),      # down
            (i, (j-1) % L),      # left
            (i, (j+1) % L)       # right
        ]
    elif lattice_type == 1: 
        # 4 cardinal neighbors
        up = ((i-1) % L, j)
        down = ((i+1) % L, j)
        left = (i, (j-1) % L)
        right = (i, (j+1) % L)
        
        if i % 2 == 1:  # Odd row
            upper_left = ((i-1) % L, (j-1) % L)
            lower_left = ((i+1) % L, (j-1) % L)
            neighbors = [up, down, left, right, upper_left, lower_left]
        else:  # Even row
            upper_right = ((i-1) % L, (j+1) % L)
            lower_right = ((i+1) % L, (j+1) % L)
            neighbors = [up, down, left, right, upper_right, lower_right]
    else: 
        raise ValueError("Lattice Type can only be 'square' (0) or 'triangular' (1)")
    
    neighbor_sum = 0.0
    for ni, nj in neighbors:
        neighbor_sum += grid[ni, nj]
    
    energy_diff = 2 * J * grid[i, j] * neighbor_sum
    
    #print('\n Calculated energy diff for spin in coordinates ', spin_coordinates, ' is ', energy_diff)
    
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
        #print('\n Energy difference is less than 0, decided TO FLIP')
        return 1
    else:
        if np.exp(-energy_diff/T)>x:
            #print('\n exp(-energy_diff/T)>x, decided TO FLIP')
            return 1
        else: 
            #print('\n exp(-energy_diff/T)<x, decided NOT TO FLIP')
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
def metropolis_sweep(grid, T, J=1.0, n_steps=5000, sample_interval=100, lattice_type=0):
    """
    Metropolis simulation with observable tracking
    
    Args:
        grid: (np.array) Initial configuration
        T: (float) Temperature
        J: (float) Coupling constant  
        n_steps: (int) Total MC steps
        sample_interval: (int) How often to record observables
        lattice_type: (int) Can be either 'square' (0) or 'triangular' (1)
        
    Returns:
        grid: (np.array) Final configuration
        energies: (np.array) Array of energy measurements
        magnetizations: (np.array) Array of magnetization measurements
    """
    
    n_samples = n_steps // sample_interval
    energies = np.zeros(n_samples)
    magnetizations = np.zeros(n_samples)
    
    sample_count = 0

    print('Starting metropolis sweep... ')
    for step in range(n_steps):
        #print('Step: ', step)
        coordinates = select_random_spin(grid)
        delta_E = calculate_energy_diff(grid, coordinates, J, lattice_type)
        if decide_flipping(delta_E, T):
            flip(grid, coordinates)
        
        # Record observables periodically
        if step % sample_interval == 0 and sample_count < n_samples:
            energies[sample_count] = calculate_total_energy(grid, J, lattice_type)
            magnetizations[sample_count] = calculate_magnetization(grid)
            sample_count += 1
    print('\nMetropolis sweep finished')
    
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

def get_critical_temperature(J=1.0, lattice_type=0):
    """
    Get theoretical critical temperature
    """
    if lattice_type == 0: #square
        return 2 * J / np.log(1 + np.sqrt(2))
    elif lattice_type == 1: #triangular
        return 4 * J / np.log(3)
    else: 
        raise ValueError("Lattice type can only be 'square' (0) or 'triangular'(1)")

def run_square_lattice(L=20, T=2.0, J=1.0, n_steps=5000):
    grid = create_random_grid((L, L))
    return metropolis_sweep(grid, T, J, lattice_type=0, n_steps=n_steps)

def run_triangular_lattice(L=20, T=3.0, J=1.0, n_steps=5000):
    grid = create_random_grid((L, L))
    return metropolis_sweep(grid, T, J, lattice_type=1, n_steps=n_steps)

def compare_lattices(L=15, T=2.5, J=1.0, n_steps=5000):
    """Compare square vs triangular lattices"""
    results = {}
    
    for lattice_type, lattice_name in [(0,"square"), (1,"triangular")]:
        print(f"Running {lattice_name} lattice...")
        
        grid = create_random_grid((L, L))
        final_grid, energies, magnetizations = metropolis_sweep(grid, T, J, n_steps, sample_interval=100, lattice_type=lattice_type)
        
        N = L * L
        results[lattice_name] = {
            'final_grid': final_grid,
            'energies': energies,
            'magnetizations': magnetizations,
            'mean_energy': np.mean(energies),
            'mean_mag': np.mean(np.abs(magnetizations)),
            'critical_temp': get_critical_temperature(J, lattice_type)
        }
    
    return results