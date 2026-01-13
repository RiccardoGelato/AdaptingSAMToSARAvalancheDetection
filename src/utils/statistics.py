import numpy as np
from scipy import stats

def calculate_correlation(x, y):
    """
    Calculates the Pearson correlation coefficient between two variables.

    Parameters:
        x (array_like): Numeric values for the first variable.
        y (array_like): Numeric values for the second variable.

    Returns:
        corr (float): Pearson correlation coefficient.
        p_value (float): Two-tailed p-value.
    """
    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Ensure both arrays have the same shape
    if x.shape != y.shape:
        raise ValueError("The input arrays must have the same shape.")
    
    # Compute Pearson's correlation coefficient and the p-value
    corr, p_value = stats.pearsonr(x, y)
    return corr, p_value