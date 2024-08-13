import numpy as np


def adjust_n(
    num_data: int,
    N: int
) -> int:
    """
    Adjust the boxcar width based on the amount of available data.
    The boxcar width is adjusted if num_data < 2N.

    Parameters
    ----------
    num_data : int
        The number of available data points.
    N : int
        The proposed boxcar width.
    
    Returns
    -------
    N : int
        The new boxcar width.
        May be equal to the original value if left unchanged.
    """

    if N % 2 == 0:
        print(f'Provided N={N} is even, proceed at your own risk.')
        # TODO: Auto adjust N if it is even? Even boxcar widths don't work well.

    if num_data < 2*N:
        N_old = N
        N = max(num_data//10, 3)
        if N % 2 == 0:
            N += 1
        print(f'Insufficient data points for given boxcar width N={N_old}. Defaulting to N={N}')

    return N


def boxcar_average(
    arr: np.ndarray,
    N: int,
    insert_val: int | float = np.nan
) -> np.ndarray:
    """
    Perform a boxcar average (AKA moving average) on the provided data.
    
    Parameters
    ----------
    arr : np.ndarray
        The input array on which the boxcar average will be performed.
    N : int
        The boxcar width.
    insert_val : some quantity
        This quantity is padded on the left and right sides of the
        averaged data so that the size of the output array
        matches the size of the input array.
    
    Returns
    -------
    bc : np.ndarray
        The array containing the averaged data.
    """

    # print(f'Applying boxcar average. Provided N={N}')
    # N = adjust_n(len(arr), N)
    if N > len(arr):
        raise ValueError(f'Provided N={N} greater than the '\
            f'number of available data points, {len(arr)}')
    
    bc = np.convolve(arr, np.ones(N)/N, mode='valid')
    for _ in range((N-1)//2):
        bc = np.insert(bc, 0, insert_val)
        bc = np.insert(bc, len(bc), insert_val)

    return bc