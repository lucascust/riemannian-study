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
                                method="iir", verbose=False).get_data()
        ext_signal.append(filtered_signal)
        
    return (np.vstack(ext_signal))


def createOnlineRaw(filtered_ext_signal, base_raw):
    info = create_info(
    ch_names=raw.ch_names,
    ch_types=['eeg'] * 8,
    sfreq=int(base_raw.info['sfreq']))
    
    return RawArray(filtered_ext_signal, info, verbose=False)

def createOfflineRaw(filtered_ext_signal, raw):
    info = create_info(
    ch_names=sum(list(map(lambda s: [ch+s for ch in raw.ch_names],
                          ["-13Hz", "-17Hz", "-21Hz"])), []),
    ch_types=['eeg'] * 24,
    sfreq=int(raw.info['sfreq']))
    
    return RawArray(filtered_ext_signal, info)
###############################################################################
# Loading EEG data

raw_fname = './data/record-[2014.03.10-20.41.35]_raw.fif'
events_fname = './data/record-[2014.03.10-20.41.35]-eve.fif'


raw, events = openEEGFile(raw_fname, events_fname)

event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
epochs = Epochs(raw, events, event_id, tmin=2, tmax=5, baseline=None)

labels = epochs.events[:, -1]

frequencies = [13., 17., 21.]
frequency_range = 0.1



raw_off_fname = './data/record-[2014.03.10-19.17.37]_raw.fif'
events_off_fname = './data/record-[2014.03.10-19.17.37]-eve.fif'

raw_off, events_off = openEEGFile(raw_off_fname, events_off_fname)

filtered_off_signal = _bandpass_filter(raw, frequencies, frequency_range)
raw_off = createOfflineRaw(filtered_off_signal, raw_off)

epochs_off = Epochs(raw_off, events_off, event_id, tmin=0, tmax=5, baseline=None)


epochs_data = epochs_off.get_data()

covariance_off = Covariances(estimator='lwf').transform(epochs_data)

mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
mdm.fit(covariance_off, labels)





iter_evoked = epochs.iter_evoked()

for evoked_number, evoked in enumerate(iter_evoked):

    
    #evoked_raw = createOnlineRaw(evoked.data, raw)

    time_1 = time.time()

    # evoked_filtered_signal = _bandpass_filter(evoked_raw, frequencies, frequency_range)
    # evoked_filtered_signal = np.array(evoked_filtered_signal)
    # evoked_filtered_signal = np.expand_dims(evoked_filtered_signal, axis=0)


    filtered_off_signal_test = evoked.data
    #filtered_off_signal_test = np.array(filtered_off_signal_test)
    filtered_off_signal_test = np.expand_dims(filtered_off_signal_test, axis=0)

    #epochs_data = np.concatenate((epochs_data, evoked_filtered_signal), axis=0)
    epochs_data = np.concatenate((epochs_data, filtered_off_signal_test), axis=0)

    cov_ext_trials = Covariances(estimator='lwf').transform(epochs_data)

    labels = np.append(labels, labels[evoked_number])

    if (evoked_number % 4 == 0 and evoked_number != 0):
        #mdm.fit(cov_ext_trials, labels)
        pass

    prediction_labeled = mdm.predict(cov_ext_trials)
    time_2 = time.time()
    prediction_labeled = prediction_labeled

    print ("Label: " + str(labels[evoked_number]))
    print ("Prediction: " + str(prediction_labeled))
    print ("Time: " + str(time_2 - time_1) + '\n')

# print("predict time = " + str(t2-t1))
# print(labels)
# print(prediction_labeled)