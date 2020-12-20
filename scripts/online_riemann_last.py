"""
Online SSVEP AI Classifier with Riemannian Geometry

- Create a machine leaning model with MDM classification
- Simulate raw EEG input and predict each trial based in offline model 
- (opitional) Retrain model for each fourth input
"""

import time
import numpy as np

from mne import get_config, set_config, find_events, read_events, create_info, Epochs
from mne.io import Raw, RawArray

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM

from sklearn.model_selection import cross_val_score, RepeatedKFold


def openEEGFile(raw_file_path, events_file_path):

    raw = Raw(raw_file_path, preload=True, verbose=False)
    events = read_events(events_file_path)

    return (raw, events)


def _bandpass_filter(signal, frequencies, freq_range):
    '''
    Filter data for any frequencies in a vector, given a frequency range
    signal: MNE raw object 
    frequencies: Vector
    '''

    ext_signal = []
    for f in frequencies:
        filtered_signal = signal.copy().filter(l_freq=f-freq_range, h_freq=f+freq_range,
                                method="iir", verbose=False).get_data()
        ext_signal.append(filtered_signal)
        
    return (np.vstack(ext_signal))



def createRaw(signal, raw, filtered):
    '''
    Create MNE Raw Object based on other Raw
    Param 1 = Data (signal)
    Param 2 = Reference Raw to extract info
    Param 3 = Is filtered? (band-pass per frequency) Default = False

    Considering Raw have N channels, if the signal from it is band-pass filtered for 3 frequencies, 
    we may have N * 3 channels in signal of input.
    '''  

    if filtered:
        ch_names = sum(list(map(lambda s: [ch+s for ch in raw.ch_names],
                        ["-13Hz", "-17Hz", "-21Hz"])), []), 
        ch_names = ch_names[0]
    else:
        ch_names = raw.ch_names

    info = create_info(
    ch_names=ch_names,
    ch_types=['eeg'] * len(ch_names),
    sfreq=int(raw.info['sfreq']))
    
    return RawArray(signal, info, verbose=False)

###############################################################################

## Option to online retraining
retrain = True

## Files to online simulation
raw_file = './data/record-[2012.07.06-19.06.14]_raw.fif'
events_file = './data/record-[2012.07.06-19.06.14]-eve.fif'

## FIles to offilne base
offline_raw_file = './data/record-[2014.03.10-20.41.35]_raw.fif'
offline_events_file = './data/record-[2014.03.10-20.41.35]-eve.fif'

# To epoching
event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
# To bandpass filtering
frequencies = [13., 17., 21.]
frequency_range = 0.1

## Loading EEG data for online prediction
raw, events = openEEGFile(raw_file, events_file)

# Filtering data for online training
signal = _bandpass_filter(raw, frequencies, frequency_range)
raw = createRaw(signal, raw, filtered=True)

epochs = Epochs(raw, events, event_id, tmin=2, tmax=5, baseline=None)
labels = epochs.events[:, -1]

## Loading EEG data for offilne ML model base
offline_raw, offline_events = openEEGFile(offline_raw_file, offline_events_file)

# Filtering data for offline training
filtered_offline_signal = _bandpass_filter(offline_raw, frequencies, frequency_range)
offline_raw = createRaw(filtered_offline_signal, offline_raw, filtered=True)

offline_epochs = Epochs(offline_raw, offline_events, event_id, tmin=2, tmax=5, baseline=None)
offline_epochs_data = offline_epochs.get_data()

# Creating ML model
offline_cov_matrix = Covariances(estimator='lwf').transform(offline_epochs_data)
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
mdm.fit(offline_cov_matrix, labels)

# Evoking trials to simulate online input
iter_evoked = epochs.iter_evoked()
epochs_data = offline_epochs_data
time_array = []

pre_predict = mdm.predict(offline_cov_matrix)

for i, evoked in enumerate(iter_evoked):

    
    evoked_raw = createRaw(evoked.data, raw, filtered=False)

    ## Start Time Counting
    time_1 = time.time()

    ## Filtering
    # evoked_filtered_signal = _bandpass_filter(evoked_raw, frequencies, frequency_range)
    # evoked_filtered_signal = np.array(evoked_filtered_signal)
    # evoked_filtered_signal = np.expand_dims(evoked_filtered_signal, axis=0)
    # epochs_data = np.concatenate((epochs_data, evoked_filtered_signal), axis=0)


    ## No Filtering
    raw_evoked_signal = evoked.data
    raw_evoked_signal = np.array(raw_evoked_signal)
    raw_evoked_signal = np.expand_dims(raw_evoked_signal, axis=0)
    epochs_data = np.concatenate((epochs_data, raw_evoked_signal), axis=0)


    cov_ext_trials = Covariances(estimator='lwf').transform(epochs_data)

    labels = np.append(labels, labels[i])

    if (i % 4 == 0 and i != 0 and retrain == True):
        mdm.fit(cov_ext_trials, labels)

    prediction_labeled = mdm.predict(cov_ext_trials)

    # Finish Time Counter    
    time_2 = time.time()
    
    time_array.append(time_2 - time_1)
    
    # print ("Predictions: ")
    # print (prediction_labeled)
    # print ("Label: " + str(labels[i]))
    # print ("Time: " + str(time_2 - time_1) + '\n')

print ("Pre-prediction: ")
print (pre_predict)

print ("Predictions: ")
print (prediction_labeled[:32])
print (prediction_labeled[32:])

mean_time = sum(time_array) / len(time_array)
print ("Mean execution time: " + str(mean_time))
