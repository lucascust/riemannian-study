# Import the necessary libraries

import numpy as np
import matplotlib.pyplot as plt

# mne import
from mne import Epochs, find_events, read_events
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
# cross validation

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

# lib to save the trained model
import pickle

# Get the dataset with the data to reproduce the real time acquisition of brain data - Pass the data for a RAW variable

raw_fname = 'data/record-[2012.07.19-11.24.02]_raw.fif'

raw = Raw(raw_fname, preload=True, verbose=False)

# find events and generate epochs

event_fname = 'data/record-[2014.03.10-19.17.37]-eve.fif'

events = read_events(event_fname)
event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
epochs = Epochs(raw, events, event_id, tmin=0, tmax=360.9, baseline=None)

# Make a pipeline with riemannian geometry models

clf = make_pipeline(Covariances(), TangentSpace(metric='riemann'), LogisticRegression())

# Get labels

labels = epochs.events[:, -1]

# Call de FIT function to rain the model

clf.fit(epochs.get_data(), labels)

# Save the model with pickle

Trained_Model = pickle.dumps(clf)

# Get the next data in "real time" and predict the label ['resting-state': 'resting', '13 Hz': 'forward', '21 Hz': 'right', '17 Hz': 'left']

Model = pickle.loads(Trained_Model)
Model.predict(raw)

# Do a clustering process and verify if the predicted label belongs to one of the clusters

#""" https://github.com/alexandrebarachant/pyRiemann/blob/d41921e201eb626992c098f126f16c75172e607c/tests/test_clustering.py """

# If the label belongs to one of the clusters, then we can save to train the new model

#""" save in a list """

# If not, do nothing, "know your place, trash"

# Verify if the list with the predicted labels equals 20
# If positive, train the new model