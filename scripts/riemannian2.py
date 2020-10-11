# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt

# MNE functions
from mne import Epochs,find_events, filter
from mne.io import Raw
from mne.decoding import Vectorizer


# Scikit-learn and Pyriemann ML functionalities
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
# mdm = pyriemann.classification.MDM()


raw = Raw("./data/subject12_run2_raw.fif", preload=True, verbose='ERROR')

print(raw)

events = find_events(raw, shortest_event=0, verbose=False)

# Pega Channels baseado no parametro
raw = raw.pick_types(eeg=True)

#Bandpass filter, passar para cada frequencia
# def _bandpass_filter(signal, lowcut, highcut):
#     """ Bandpass filter using MNE """
#     return signal.copy().filter(l_freq=lowcut, h_freq=highcut,
#                                 method="iir").get_data()


# Plot MNE
raw.plot(duration=2, start=0, n_channels=8, scalings={'eeg': 4e-2},
         color={'eeg': 'steelblue'})
plt.show()

event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
epochs = Epochs(raw, events, event_id, tmin=2, tmax=5, baseline=None)

cov_raw = Covariances(estimator='lwf').transform(epochs.get_data())

# Verificar necessidade de tirar as medias no fim


# from sklearn.linear_model import LogisticRegression
# var = epochs.get_data()
# cov_raw = scikitlearn.pipeline(Covariances(estimator='lwf').transform(var), TangentSpace().transform(epochs.get_data(var), logisticRegression.fit(x,y,weight))

#Tangent 


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