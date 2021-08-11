"""
====================================================================
Offline SSVEP-based BCI Multiclass Prediction
====================================================================
"""

# generic import
import os
import numpy as np

# mne import
from mne import get_config, set_config, find_events, read_events, create_info, Epochs
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM

# scikit-learn import
from sklearn.model_selection import cross_val_score, RepeatedKFold

# lib to save the trained model
import pickle

import time

# Open EEG file given raw and event paths
def openEEGFile(raw_file_path, events_file_path):

    raw = Raw(raw_file_path, preload=True, verbose=False)
    events = read_events(events_file_path)

    return (raw, events)


# Filter data for any frequencies in a vector, given a frequency range
# signal: MNE raw object 
# frequencies: Vector
def _bandpass_filter(signal, frequencies, freq_range):
    """ Bandpass filter using MNE """
    ext_signal = []
    for f in frequencies:
        filtered_signal = signal.copy().filter(l_freq=f-freq_range, h_freq=f+freq_range,
                                method="iir").get_data()
        ext_signal.append(filtered_signal)
        
    return (np.vstack(ext_signal))


def createNewRaw(filtered_ext_signal, raw):
    info = create_info(
    ch_names=sum(list(map(lambda s: [ch+s for ch in raw.ch_names],
                          ["-13Hz", "-17Hz", "-21Hz"])), []),
    ch_types=['eeg'] * 24,
    sfreq=int(raw.info['sfreq']))
    
    return RawArray(filtered_ext_signal, info)
###############################################################################
# Loading EEG data

raw_fname = './data/old_data/record-[2014.03.10-20.41.35]_raw.fif'
events_fname = './data/old_data/record-[2014.03.10-20.41.35]-eve.fif'

raw, events = openEEGFile(raw_fname, events_fname)

###############################################################################


frequencies = [13., 17., 21.]
frequency_range = 0.1

ext_signal = _bandpass_filter(raw, frequencies, frequency_range)

###############################################################################

raw_ext = createNewRaw(ext_signal, raw)

###############################################################################

event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}

epochs = Epochs(raw_ext, events, event_id, tmin=2, tmax=5, baseline=None)

t1 = time.time()
cov_ext_trials = Covariances(estimator='lwf').transform(epochs.get_data())

###############################################################################

# Get labels

labels = epochs.events[:, -1]

mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
mdm.fit(cov_ext_trials, labels)
t2 = time.time()

prediction_labeled = mdm.predict(cov_ext_trials)


print("predict time = " + str(t2-t1))
print(labels)
print(prediction_labeled)