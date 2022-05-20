import numpy as np


def adjust_n(num_data, N):
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

    if num_data < 2*N:
        N_old = N
        N = max(num_data//10, 3)
        if N%2 == 0:
            N += 1
        print('Insufficient data points for given boxcar width N=' + str(N_old) + '. Defaulting to N=' + str(N))

    return N


def boxcar_average(arr, N, insert_val=np.nan):
    """
    Perform a boxcar (or moving) average on the provided data.
    N is automatically adjusted if the proposed value is too large.
    NOTE: For AIA data, N = 51 corresponds to a boxcar average
    over 10 minutes and 12 seconds (even boxcars are not as nice).
    
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
    N : int
        The boxcar width used to perform the average.
    """

    print('Applying boxcar average. Provided N=' + str(N))
    N = adjust_n(len(arr), N)

    bc = np.convolve(arr, np.ones(N)/N, mode='valid')
    for i in range((N-1)//2):
        bc = np.insert(bc, 0, insert_val)
        bc = np.insert(bc, len(bc), insert_val)

    return bc, N