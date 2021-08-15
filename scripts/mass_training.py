"""
====================================================================
Offline SSVEP-based BCI Multiclass Prediction
====================================================================
"""

# generic import
import os
import numpy as np

# mne import
from mne import get_config, set_config, find_events, read_events, create_info, Epochs
from mne.io import Raw, RawArray

# pyriemann import
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# scikit-learn import
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

import moabb
from moabb.datasets import SSVEPExo
from moabb.evaluations import CrossSubjectEvaluation
from moabb.paradigms import SSVEP, FilterBankSSVEP
from moabb.pipelines import SSVEP_CCA, ExtendedSSVEPSignal

import time

dataset = SSVEPExo()
interval = dataset.interval


paradigm_fb = FilterBankSSVEP(filters=None, n_classes=4)


filtered_db = paradigm_fb.get_data(dataset, return_epochs=False)

pipelines_fb = {}
pipelines_fb["RG+LogReg"] = make_pipeline(
    ExtendedSSVEPSignal(),
    Covariances(estimator="lwf"),
    TangentSpace(),
    LogisticRegression(solver="lbfgs", multi_class="auto"),
)

overwrite = False  # set to True if we want to overwrite cached results

evaluation_fb = CrossSubjectEvaluation(
    paradigm=paradigm_fb, datasets=dataset, overwrite=overwrite
)
results_fb = evaluation_fb.process(pipelines_fb)

print(results_fb['score'])