import numpy as np

def smape(pred, data):
    true = data.get_label()
    return 'smape', np.mean((np.abs(true - pred)) / (np.abs(true) + np.abs(pred))) * 100, False
