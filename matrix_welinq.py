from numpy import array, ix_, zeros
from numpy.linalg import norm
"example of matrix definition for welinq"

def get_stiffness(x_size: int, y_size: int, delta_x: float, delta_y: float, alpha: int) -> array:
    """
    Returns the stiffness matrix for a 2D grid.

    Parameters:
        x_size (int): the number of grid points along the x-axis.
        y_size (int): the number of grid points along the y-axis.
        delta_x (float): the grid spacing in the x-direction.
        delta_y (float): the grid spacing in the y-direction.
        alpha (int): a constant factor that influences the stiffness.

    Returns:
        lil_matrix: a sparse matrix representing the stiffness of the grid.
    """
    size = x_size * y_size # Number of grid points
    matrix = zeros((size, size))
    values = (alpha / delta ** 2 for delta in (delta_x, delta_y))
    submatrices = [array([[value, -value], [-value, value]], dtype=float) 
                   for value in values]
    
    for i, j in ((i, j) for i in range(x_size) for j in range(y_size)):
        index = i * y_size + j # Current grid point
        right = index + y_size # Neighboring points
        above = index + 1

        if i < x_size - 1: # Update the a 2x2 submatrix of the stiffness matrix
            matrix[ix_([index, right], [index, right])] += submatrices[0]

        if j < y_size - 1: # Update the a 2x2 submatrix of the stiffness matrix
            matrix[ix_([index, above], [index, above])] += submatrices[1]

    return matrix

def get_force(x_size: int, y_size: int, percentage: float, force_value: float) -> array:
    """ 
    Constructs, normalizes and returns a force vector for a system, where the force is applied 
    to a specified percentage of the grid area.

    Parameters:
        x_size (int): the number of grid points along the x-axis.
        y_size (int): the number of grid points along the y-axis.
        percentage (float): percentage of the grid's x-size, from the right side, where the force is applied.
        force_value (float): magnitude of the force to be applied to the specified region of the grid.

    Returns:
        array: a normalized force vector.
    """
    force = zeros(x_size * y_size) # Degrees of freedom
    boundary = round((1 - percentage) * y_size) # Starting point of the area where force is applied
    
    for i in range(boundary, y_size):
        force[(x_size - 1) * y_size + i] = force_value
        
    force /= norm(force)
    
    return force