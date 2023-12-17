import numpy as np

def SMAPE(true, pred):
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 100
