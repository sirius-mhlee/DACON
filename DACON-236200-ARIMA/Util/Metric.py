from sklearn.metrics import mean_absolute_error

def mae(true, pred):
    return mean_absolute_error(true, pred)
