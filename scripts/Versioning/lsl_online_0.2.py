"""
Online SSVEP AI Classifier with Riemannian Geometry

- Create a machine leaning model with MDM classification
- Get data via LSL
- The training inputs labels (not realistic simulation)
- (optional) Retrain model for each fourth input
"""

import time
import numpy as np
import keyboard
import copy

from pylsl import StreamInlet, resolve_stream

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


## FIles to offilne base
offline_raw_file = './data/record-[2012.07.06-19.06.14]_raw.fif'
offline_events_file = './data/record-[2012.07.06-19.06.14]-eve.fif'

# To epoching
event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
# To bandpass filtering
frequencies = [13., 17., 21.]
frequency_range = 0.1


## Loading EEG data for offilne ML model base
offline_raw, offline_events = openEEGFile(offline_raw_file, offline_events_file)
raw = copy.deepcopy(offline_raw)
# Filtering data for offline training
filtered_offline_signal = _bandpass_filter(offline_raw, frequencies, frequency_range)
offline_raw = createRaw(filtered_offline_signal, offline_raw, filtered=True)

offline_epochs = Epochs(offline_raw, offline_events, event_id, tmin=2, tmax=5, baseline=None)
offline_epochs_data = offline_epochs.get_data()
labels = offline_epochs.events[:, -1]
# Creating ML model
offline_cov_matrix = Covariances(estimator='lwf').transform(offline_epochs_data)
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
mdm.fit(offline_cov_matrix, labels)

# Fim do Offline

epochs_data = copy.deepcopy(offline_epochs_data)
time_array = []

print("\nlabels: ")
print(labels, "\n")

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('name', 'openvibeSignal')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

### Test Area
t = time.time()

## Teste: Tempo entre cada timestamp e cada coleta de dado
# sample1, timestamp1 = inlet.pull_sample()
# sample2, timestamp2 = inlet.pull_sample()
# print(time.time_ns())
# print(timestamp1)
# print(timestamp2)
## Resultado: 0,00390625s (1/256)

## Teste: Dados introduzidos no vetor após 1s de execução
arr=[]
while time.time() - t < 1:
    sample, timestamp = inlet.pull_sample()
    print(inlet.samples_available())
    arr.append(sample)
print("depois de 1s:")
print(len(arr))
print(arr)
## Resultado: 257

###

sample, timestamp = inlet.pull_sample()
time_window = np.array(sample)


sample, timestamp = inlet.pull_sample()
sample = np.array(sample)
time_window = np.column_stack((sample, time_window))
time_window_base = time_window

count = 0

timeBase = time.time()

while not keyboard.is_pressed('s'):
    time_window = time_window_base
    while time_window.shape[1] < 769:
        sample, timestamp = inlet.pull_sample()
        sample = np.array(sample)
        time_window = np.column_stack((time_window, sample))

    actualTime = time.time() - timeBase
    print("Tempo atual: " + str(actualTime))
    print("timestamp: " + str(timestamp))
    print("deltaT = " + str(timestamp - actualTime) + "\n")

        
    signal_raw = createRaw(time_window, raw, filtered=False)

    ## Start Time Counting
    time_1 = time.time()

    ## Filtering
    filtered_signal = _bandpass_filter(signal_raw, frequencies, frequency_range)
    filtered_signal = np.array(filtered_signal)
    filtered_signal = np.expand_dims(filtered_signal, axis=0)
    epochs_data = np.concatenate((epochs_data, filtered_signal), axis=0)


    ## No Filtering
    # raw_evoked_signal = evoked.data
    # raw_evoked_signal = np.array(raw_evoked_signal)
    # raw_evoked_signal = np.expand_dims(raw_evoked_signal, axis=0)

    cov_ext_trials = Covariances(estimator='lwf').transform(epochs_data)

    labels = np.append(labels, labels[count])

    # Fit a cada 4 janelas de tempo
    if (count % 4 == 0 and count != 0 and retrain == True):
        mdm.fit(cov_ext_trials, labels)
        print("RETRAINED")

    prediction_labeled = mdm.predict(cov_ext_trials)

    # Finish Time Counter    
    time_2 = time.time()

    time_array.append(time_2 - time_1)
    count+=1
    print ("Predictions: ")
    print (prediction_labeled[:32])
    print (prediction_labeled[32:])
    # print ("Label: " + str(labels[i]))
    print ("Time: " + str(time_2 - time_1) + '\n')



print ("Predictions: ")
print (prediction_labeled[:32])
print (prediction_labeled[32:])

mean_time = sum(time_array) / len(time_array)
print ("Mean execution time: " + str(mean_time))
