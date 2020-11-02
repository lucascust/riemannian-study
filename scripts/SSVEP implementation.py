# Import the necessary libraries

import numpy as np
import matplotlib.pyplot as plt

# mne import
from mne import Epochs, find_events, read_events, pick_types
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
# cross validation

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# lib to save the trained model
import pickle

# Get the dataset with the data to reproduce the real time acquisition of brain data - Pass the data for a RAW variable
raw_fname = '../data/record-[2012.07.19-11.24.02]_raw.fif'
raw = Raw(raw_fname, preload=True, verbose=False)

# replace baselining with high-pass
raw.filter(2, None, method='iir') 

# find events and generate epochs

event_fname = '../data/record-[2014.03.10-19.17.37]-eve.fif'
events = read_events(event_fname)
event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}

## Modificação de parâmetros Epoch e adicionado "Picks"
# Define time range (1 to labeled)
# Only works up to 151s in record-[2012.07.19-11.24.02]
tmin, tmax = -0., 1

# epochs = Epochs(raw, events, event_id, tmin=0, tmax=360.9, baseline=None)
epochs = Epochs(raw, events, event_id, tmin, tmax,  proj=False, baseline=None, preload=True, verbose=False)

# Extract data from Epochs object in (Trial, Channel, Sample) format
epochs_data = epochs.get_data()

# Make a pipeline with riemannian geometry models

clf = make_pipeline(Covariances(), TangentSpace(metric='riemann'), LogisticRegression())

# Get labels

labels = epochs.events[:, -1]

# Call de FIT function to train the model

clf.fit(epochs_data, labels)

# Save the model with pickle

Trained_Model = pickle.dumps(clf)

# Get the next data in "real time" and predict the label ['resting-state': 'resting', '13 Hz': 'forward', '21 Hz': 'right', '17 Hz': 'left']

Model = pickle.loads(Trained_Model)
# Transpoes matriz 3d to put samples in first index
epochs_data_transposed = epochs_data.transpose(2,1,0)
# Predict all samples
prediction = Model.predict(epochs_data_transposed)
# Predict only labeled samples (use 1s of time range)
prediction_labeled = Model.predict(epochs_data)

# Tentativa de transformar as Epochs em matriz de covariancia para fazer o predict com dados de mesmo escopo
# cov = Covariances().transform(epochs_data)
# prediction = Model.predict(cov)