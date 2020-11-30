def butter_bandpass(signal, lowcut, highcut, fs=254, order=1, filttype='forward-backward'):
    signal = signal.get_data()
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if filttype == 'forward':
        pass
    elif filttype == 'forward-backward':
        filtered = filtfilt(b, a, signal, axis=-1)
    else:
        raise ValueError("Unknown filttype:", filttype)    
    return filtered
