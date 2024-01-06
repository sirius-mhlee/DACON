import numpy as np

def smape(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    return np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 100
