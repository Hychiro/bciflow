'''
Description
-----------
This module implements a convolution-based bandpass filter for EEG data. 
The `bandpass_conv` function uses a kernel derived from windowed sinc 
functions to perform the filtering.

Function
------------
'''
import numpy as np

def bandpass_conv(eegdata, low_cut=4, high_cut=40, transition=None, window_type='hamming', kind='same', inplace=False):
    '''
    Parameters
    ----------
    eegdata : dict
        A dictionary containing the EEG data, where the key 'X' holds the 
        raw signal and 'sfreq' holds the sampling frequency.
    low_cut : int
        The lower cutoff frequency of the bandpass filter (default is 4 Hz).
    high_cut : int
        The upper cutoff frequency of the bandpass filter (default is 40 Hz).
    transition : int or float
        The transition bandwidth for the filter (default is half the passband width).
    window_type : str
        The type of window used for the sinc function ('hamming' or 'blackman', default is 'hamming').
    kind : str
        The convolution mode ('same' or 'valid', default is 'same').

    Returns
    -------
    output : dict
        The original dictionary with the filtered data stored under the key 'X'.
    '''

    # -------- Validate eegdata --------
    if not isinstance(eegdata, dict):
        raise ValueError("eegdata must be a dictionary.")

    if 'X' not in eegdata:
        raise ValueError("eegdata must contain the key 'X'.")

    if 'sfreq' not in eegdata:
        raise ValueError("eegdata must contain the key 'sfreq'.")

    X = eegdata['X']
    sfreq = eegdata['sfreq']

    # -------- Validate X --------
    if not isinstance(X, np.ndarray):
        raise ValueError("eegdata['X'] must be a numpy array.")

    if X.ndim < 2:
        raise ValueError("eegdata['X'] must have at least 2 dimensions.")

    if X.shape[-1] <= 1:
        raise ValueError("Signal must contain more than one sample.")

    # -------- Validate sfreq --------
    if not isinstance(sfreq, (int, float)):
        raise ValueError("sfreq must be numeric.")

    if sfreq <= 0:
        raise ValueError("sfreq must be greater than zero.")

    nyquist = sfreq / 2

    # -------- Validate cutoff frequencies --------
    if not isinstance(low_cut, (int, float)):
        raise ValueError("low_cut must be numeric.")

    if not isinstance(high_cut, (int, float)):
        raise ValueError("high_cut must be numeric.")

    if low_cut <= 0 or high_cut <= 0:
        raise ValueError("Cutoff frequencies must be greater than zero.")

    if low_cut >= high_cut:
        raise ValueError("low_cut must be smaller than high_cut.")

    if high_cut >= nyquist:
        raise ValueError("high_cut must be smaller than the Nyquist frequency.")


    # -------- Validate transition --------
    if transition is None:
        transition = (high_cut - low_cut) / 2

    if isinstance(transition, (int, float)):
        if transition <= 0:
            raise ValueError("transition must be greater than zero.")
        transition = [float(transition), float(transition)]

    elif isinstance(transition, (list, tuple)):
        if len(transition) != 2:
            raise ValueError("transition must contain two values.")
        if transition[0] <= 0 or transition[1] <= 0:
            raise ValueError("transition values must be greater than zero.")
    else:
        raise ValueError("transition must be numeric or a list/tuple of two numbers.")

    # -------- Validate window --------
    valid_windows = ['hamming', 'blackman']
    if window_type not in valid_windows:
        raise ValueError(f"window_type must be one of {valid_windows}.")

    # -------- Validate convolution mode --------
    valid_modes = ['same', 'valid']
    if kind not in valid_modes:
        raise ValueError(f"kind must be one of {valid_modes}.")
    
    # -------- Copy logic --------
    if not inplace:
        eegdata = eegdata.copy()

    X = eegdata['X'].copy()
    sfreq = eegdata['sfreq']
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))

    NL = int(4 * sfreq / transition[0])
    NH = int(4 * sfreq / transition[1])


    hlpf = np.sinc(2 * high_cut / sfreq * (np.arange(NH) - (NH - 1) / 2))
    if window_type=='hamming':
        hlpf *= np.hamming(NH)
    elif window_type=='blackman':
        hlpf *= np.blackman(NH)
    hlpf /= np.sum(hlpf)

    hhpf = np.sinc(2 * low_cut / sfreq * (np.arange(NL) - (NL - 1) / 2))
    if window_type=='hamming':
        hhpf *= np.hamming(NL)
    elif window_type=='blackman':
        hhpf *= np.blackman(NL)
    hhpf /= np.sum(hhpf)
    hhpf = -hhpf
    hhpf[(NL - 1) // 2] += 1

    kernel = np.convolve(hlpf, hhpf)
    if len(kernel) > X.shape[-1] and kind == 'same':
        kind = 'valid'

    X_ = []
    for signal_ in range(X.shape[0]):
        filtered = np.convolve(X[signal_], kernel, mode=kind)
        X_.append(filtered)

    X_ = np.array(X_)
    X_ = X_.reshape(eegdata['X'].shape)
    eegdata['X'] = X_
    return eegdata