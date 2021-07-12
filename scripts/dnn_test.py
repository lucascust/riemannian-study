from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Dropout
import numpy as np

import numpy as np

# mne import
from mne import get_config, set_config, find_events, read_events, create_info, Epochs
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.classification import MDM

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

raw_fname = './data/record-[2014.03.10-20.41.35]_raw.fif'
events_fname = './data/record-[2014.03.10-20.41.35]-eve.fif'

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

cov_ext_trials = Covariances(estimator='lwf').transform(epochs.get_data())

# Get labels

labels = epochs.events[:, -1]
labels = labels - 1

n_classes=4
sample_shape = (576,) #24*24
epochs = 100


matrix_size = 576 

# Matrix to vector
X_train = cov_ext_trials[:24]
X_train = X_train.reshape(X_train.shape[0], matrix_size)

X_test = cov_ext_trials[24:]
X_test = X_test.reshape(X_test.shape[0], matrix_size)

y_train = to_categorical(labels[:24], n_classes)
y_test = to_categorical(labels[24:], n_classes)

# Create the model
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=sample_shape))
model.add(Dropout(0.2))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    verbose=True,
                    validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Acurácia do teste:', score[1])