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
from pyriemann.classification import TangentSpace
from pyriemann.clustering import Kmeans

import moabb
from moabb.datasets import SSVEPExo
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import SSVEP, FilterBankSSVEP
from moabb.pipelines import SSVEP_CCA, ExtendedSSVEPSignal

# scikit-learn import
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

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
dataset = SSVEPExo(subjects=11)
interval = dataset.interval


paradigm_fb = FilterBankSSVEP(filters=None, n_classes=4)


filtered_db = paradigm_fb.get_data(dataset, return_epochs=False)

pipeline = make_pipeline(
    ExtendedSSVEPSignal(),
    Covariances(estimator="lwf"),
    TangentSpace(),
    LogisticRegression(solver="lbfgs", multi_class="auto"),
)

model = pipeline.fit()

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


    #### 
    # Guardar dado num buffer de dados

    # predizer ele com o modelo offline
    
    # Guarda no buffer de labels
    #####


    ### Recalibração esporádica parela
    # depois de muito tempo (em paralelo), pegar o buffer de dados , fazer clusterização
    # pegar novas labels 

    # fazer votação entre as labels do offlne (tgspace, logreg) com as labels do online (kmeans)

    # usar resultado e passar para o modelo offline retreinar do zero


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
