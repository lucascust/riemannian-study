import pyriemann
from sklearn.model_selection import cross_val_score

# load your data
X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
y = ... # the labels

# estimate covariances matrices
cov = pyriemann.estimation.Covariances().fit_transform(X)

# cross validation
mdm = pyriemann.classification.MDM()

accuracy = cross_val_score(mdm, cov, y)

print(accuracy.mean())