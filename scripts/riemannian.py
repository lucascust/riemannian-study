import numpy as np
import matplotlib.pyplot as plt

# mne import
from mne import Epochs, find_events
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
# cross validation

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold



raw = Raw("./data/subject9_2.fif", preload=True, verbose='ERROR')
print(raw)
events = find_events(raw)
print (events)
# Pega Channels baseado no parametro
#raw = raw.pick_types(eeg=True)

#Bandpass filter


# Plot MNE
# raw.plot(duration=2, start=0, n_channels=8, scbalings={'eeg': 4e-2},
#          color={'eeg': 'steelblue'})
# plt.show()

event_id = {'13 Hz': 2, '17 Hz': 4, '21 Hz': 3, 'resting-state': 1}
epochs = Epochs(raw, events, event_id, tmin=0, tmax=360.9, baseline=None)

# cov_raw = Covariances(estimator='lwf').transform(epochs.get_data())
# cov_ts = TangentSpace().fit_transform(cov_raw)
# print(cov_ts)

labels = epochs.events[:, -1]
print(epochs)
print(epochs.events)
preds = np.zeros(len(labels))


epochs_data = epochs.get_data()

# Define a monte-carlo cross-validation generator (reduce variance):
cv = KFold(n_splits=10, shuffle=True, random_state=42)

clf = make_pipeline(Covariances(), TangentSpace(metric='riemann'), LogisticRegression())

for train_idx, test_idx in cv.split(epochs_data):
    y_train, y_test = labels[train_idx], labels[test_idx]
    clf.fit(epochs_data[:28], y_train[:min(28,len(y_train))])
    preds[test_idx] = clf.predict(epochs_data[test_idx])


# clf.fit(epochs.get_data(), labels)
# clf.predict(events)

# Printing the results
acc = np.mean(preds == labels)
print("Classification accuracy: %f " % (acc))
names = ['resting-state', '13 Hz', '21 Hz', '17 Hz']
plot_confusion_matrix(preds, labels, names)
plt.show()

#from sklearn.linear_model import LogisticRegression
#var = epochs.get_data()
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