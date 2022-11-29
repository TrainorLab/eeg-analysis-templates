import numpy as np
import scipy as sp


def run_ab(eeg, threshold, method='window'):
    """
    Run artifact blocking on data in an MNE Raw object. Modifies
    the EEG data in-place. Note that the EEG data is expected to
    be in volts (MNE's standard scaling), while the artifact 
    threshold is expected to be in microvolts. 
    
    :param eeg: An MNE Raw object containing EEG data (in V) as
        channels x time.
    :param threshold: The amplitude threshold (in uV) above which
        activity will be treated as artifactual.
    :param method: If 'total', performs artifact blocking across
        the entire session in one step. If 'window', performs
        artifact blocking on windows of data across the session.
        
    return: None (Modifies the EEG data in place.)
    """
    
    # Define window size based on specified approach
    if len(eeg._data.shape) != 2:
        raise ValueError('EEG data must be arranged as channels x time. The data provided was of shape %s.' % eeg._data.shape)
    nchans, ntimes = eeg._data.shape
    if method == 'total':
        lwin = ntimes
    elif method == 'window':
        lwin = int(min([ntimes, 10 * eeg.info['sfreq']]))
    else:
        raise ValueError('Method must be "total" or "window", not "%s".' % method)
    
    # Convert threshold from uV to V, since MNE keeps data in V
    threshold /= 1000000
    
    # Create copy of data with mean removed 
    old_eeg = eeg.copy()
    ch_means = np.mean(old_eeg._data, axis=1)
    for i, m in enumerate(ch_means):
        old_eeg._data[i, :] -= m
    
    ##########
    #
    # Artifact Blocking (Total)
    #
    ##########
    
    if lwin == ntimes:
        
        # Zero out artifactual samples
        bad = np.abs(old_eeg._data) > threshold
        eeg._data[bad] = 0
        
        # Mix original signals (with means removed) with cleaned signals
        rxx = np.dot(old_eeg._data, old_eeg._data.T)
        ryx = np.dot(eeg._data, old_eeg._data.T)
        W = np.dot(ryx, np.linalg.pinv(rxx))
        eeg._data = W * old_eeg._data

    ##########
    #
    # Artifact Blocking (Windowed)
    #
    ##########
    
    else:
        
        eeg._data.fill(0)
        start = 0
        end = lwin
        r = .02
        finished = False
        
        while not finished:
            
            # Create window
            win_tail = 'twotailed'
            lwin2 = lwin
            if start == 0:
                win_tail = 'right'
            if end > ntimes:
                win_tail = 'left'
                end = ntimes
                lwin2 = end - start
            win = tukey_window(lwin, r, win_tail, lwin2)
            
            # Apply window to data
            win_data = np.zeros((nchans, len(win)))
            for i in range(nchans):
                win_data[i, :] = win * old_eeg._data[i, start:end]
            
            # Perform artifact blocking on window of data
            if win_data.shape[1] > win_data.shape[0]:
                bad = win_data > threshold
                win_data_clean = win_data.copy()
                win_data_clean[bad] = 0
                rxx = np.dot(win_data, win_data.T)
                ryx = np.dot(win_data_clean, win_data.T)
                W = np.dot(ryx, np.linalg.pinv(rxx))
                eeg._data[:, start:end] += np.dot(W, win_data)
            
            # Move window forward unless we have reached the end
            if end >= ntimes:
                finished = True    
            else:
                start = end - int(np.fix(r * lwin / 4.))
                end = start + lwin


def tukey_window(n, r, tail='twotailed', n2=None):
    """
    Generates a Tukey window.
    
    :param n: The length of the window.
    :param r: The fraction of the window inside the cosine tapered region.
    :param tail: A string indicating whether the window should have two tails
        ('twotailed'), only a leading tail ('left'), or only a trailing tail ('right').
    :param n2: If using a leading tail, the returned window will be the first
        n2 points of a length-n tukey window.
    
    return: A 1D numpy array of length n containing the generated Tukey window.
    """
    # This function generates tukeywin window of length N and ratio r.
    if n2 is None:
        n2 = n
    elif n2 > n:
        raise ValueError('Value of n2 cannot be greater than n.')
    
    # Create one-sided Tukey window with only a trailing tail
    if tail == 'right':
        win = sp.signal.tukey(n ,r)
        win[:int(np.fix(r * n / 2.))] = 1
    # Create standard Tukey window
    elif tail == 'twotailed':
        win = sp.signal.tukey(n, r)
    # Create one-sided Tukey window with only a leading tail
    elif tail == 'left':
        win = sp.signal.tukey(n, r)[:n2]
    # Other indexes not recognized
    else:
        raise ValueError('Value of tail not recognized. Must be "twotailed", "leading", or "trailing".')
        
    return win