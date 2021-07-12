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
from pyriemann.clustering import Kmeans

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

def timeWindowInit(inlet):

    sample, timestamp = inlet.pull_sample()
    time_window = np.array(sample)

    sample, timestamp = inlet.pull_sample()
    sample = np.array(sample)
    return np.column_stack((sample, time_window))


###############################################################################
## Option to online retraining
retrain = True


## FIles to offilne base
off_raw_file = './data/record-[2012.07.06-19.06.14]_raw.fif'
off_events_file = './data/record-[2012.07.06-19.06.14]-eve.fif'

# To epoching
event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
# To bandpass filtering
frequencies = [13., 17., 21.]
frequency_range = 0.1


## Loading EEG data for offilne ML model base
off_raw, off_events = openEEGFile(off_raw_file, off_events_file)
raw = copy.deepcopy(off_raw)
# Filtering data for offline training
filtered_off_signal = _bandpass_filter(off_raw, frequencies, frequency_range)
# Create new MNE Raw with filtered data
off_raw = createRaw(filtered_off_signal, off_raw, filtered=True)
# Get Epochs
off_epochs = Epochs(off_raw, off_events, event_id, tmin=2, tmax=5, baseline=None)
# Get Epochs data (signal)
off_epochs_data = off_epochs.get_data()
epochs_data = copy.deepcopy(off_epochs_data)
# Get Labels
labels = off_epochs.events[:, -1]
labels_base = copy.deepcopy(labels)
# Covariance Matrix transorm
off_cov_matrix = Covariances(estimator='lwf').transform(off_epochs_data)
# MDM model init and fit
mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
mdm.fit(off_cov_matrix, labels)

# End of offline training

# EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('name', 'openvibeSignal')
# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

kmeans = Kmeans(n_clusters=4)

time_window = timeWindowInit(inlet)
time_window_base = copy.deepcopy(time_window)
timeBase = time.time()

count = 0
time_array = []
while not keyboard.is_pressed('s'):
    time_window = copy.deepcopy(time_window_base)
    while time_window.shape[1] < 769:
        sample, timestamp = inlet.pull_sample()
        sample = np.array(sample)
        time_window = np.column_stack((time_window, sample))

    actualTime = time.time() - timeBase
    print("Runing time: " + str(actualTime))
    print("timestamp: " + str(timestamp))
    print("deltaT = " + str(timestamp - actualTime) + "\n")

        
    signal_raw = createRaw(time_window, raw, filtered=False)

    ## Start execution time counting 
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


    ### TREINO ONLINE: AQUI NÃO FAZ SENTIDO
    #labels = np.append(labels, labels[count])

    # Fit a cada 4 janelas de tempo
    if (count % 4 == 0 and count != 0 and retrain == True):
        kmeans.fit(cov_ext_trials)
        print("RETRAINED")

    if (count > 5):
        prediction_labeled = kmeans.predict(cov_ext_trials)
    else:
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

    if count == 7:
        print ("FIRST TEST")
        print (labels_base)
        results = [str(x+1) for x in prediction_labeled[:32]]
        print (results)


print ("Predictions: ")
print (prediction_labeled[:32])
print (prediction_labeled[32:])

mean_time = sum(time_array) / len(time_array)
print ("Mean execution time: " + str(mean_time))
