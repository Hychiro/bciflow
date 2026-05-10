'''
Description
-----------
This module implements a Chebyshev Type II bandpass filter for EEG data. 
The `chebyshevII` function applies a recursive filter with a steeper roll-off 
and controlled stopband ripple.

Function
------------
'''
import numpy as np
from scipy.signal import cheby2, filtfilt

def chebyshevII(eegdata, low_cut=4, high_cut=40, btype='bandpass', order=4, rs='auto', inplace=False):
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
    kind_bp : str
        The type of filter ('bandpass', 'lowpass', 'highpass', etc., default is 'bandpass').
    order : int
        The order of the filter (default is 4).
    rs : str
        The minimum attenuation in the stopband (default is 'auto', 
        which sets 40 dB for bandpass and 20 dB for other types).

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
        raise ValueError("The signal must contain more than one sample.")

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

    # -------- Validate filter type --------
    valid_btypes = ['bandpass', 'lowpass', 'highpass', 'bandstop']
    if btype not in valid_btypes:
        raise ValueError(f"btype must be one of {valid_btypes}.")

    # -------- Validate order --------
    if not isinstance(order, int):
        raise ValueError("order must be an integer.")

    if order <= 0:
        raise ValueError("order must be greater than zero.")

    # -------- Validate rs --------
    if rs == 'auto':
        rs = 40 if btype == 'bandpass' else 20
    else:
        if not isinstance(rs, (int, float)):
            raise ValueError("rs must be numeric or 'auto'.")
        if rs <= 0:
            raise ValueError("rs must be greater than zero.")
        
    if inplace:
        output = eegdata
    else:
        output = eegdata.copy()
        output['X'] = eegdata['X'].copy()    
        
    if btype in ['bandpass', 'bandstop']:
        Wn = [low_cut, high_cut]
    elif btype == 'lowpass':
        Wn = high_cut
    elif btype == 'highpass':
        Wn = low_cut

    X = output['X'].copy()
    X = X.reshape((np.prod(X.shape[:-1]), X.shape[-1]))
    
    b, a = cheby2(order, rs, Wn, btype=btype, fs=sfreq)
    X_ = []
    for signal_ in range(X.shape[0]):

        filtered = filtfilt(b, a, X[signal_])
        X_.append(filtered)

    X_ = np.array(X_)
    X_ = X_.reshape(output['X'].shape)
    output['X'] = X_

    return output