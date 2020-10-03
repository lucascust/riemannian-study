import numpy as np
import matplotlib.pyplot as plt

# mne import
from mne import Epochs, find_events
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances



raw = Raw("./data/subject12_run2_raw.fif", preload=True, verbose='ERROR')

events = find_events(raw, shortest_event=0, verbose=False)

# Pega Channels baseado no parametro
raw = raw.pick_types(eeg=True)



# Plot MNE
raw.plot(duration=2, start=0, n_channels=8, scalings={'eeg': 4e-2},
         color={'eeg': 'steelblue'})
plt.show()

event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
epochs = Epochs(raw, events, event_id, tmin=2, tmax=5, baseline=None)

cov_raw = Covariances(estimator='lwf').transform(epochs.get_data())


print("stop")
# load your data
# X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
# y = ... # the labels

# # estimate covariances matrices
# cov = pyriemann.estimation.Covariances().fit_transform(X)

# # cross validation
# mdm = pyriemann.classification.MDM()



# accuracy = cross_val_score(mdm, cov, y)

# print(accuracy.mean())