import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import statsmodels.tsa.arima.model as smt

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Util.Preprocessing import date_preprocessing
from Util.Metric import mae

def main():
    # Data Load
    df = pd.read_csv('./Data/train.csv')
    if Config.test_run:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    print()
    print(df.info())

    # Data Preprocessing
    date_preprocessing(df)

    print()
    print(df.info())

    # Define Train Valid
    split_train_idx, split_val_idx = train_test_split(np.arange(len(df)), test_size=0.2, shuffle=False)

    # Train
    fold_score_list = []

    data_generator = [(split_train_idx, split_val_idx)]

    print()
    for fold, (train_idx, val_idx) in enumerate(data_generator):
        fold += 1

        print('Fold: {}'.format(fold))

        fold_train = df.iloc[train_idx, :]
        fold_val = df.iloc[val_idx, :]

        print()

        model = smt.ARIMA(fold_train['평균기온'], order=(1, 1, 0), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit()
        print(model_fit.summary())

        print()

        fold_pred = model_fit.predict(start=fold_val.index.min(), end=fold_val.index.max())
        fold_score = mae(fold_val['평균기온'], fold_pred.values)

        fold_score_list.append(fold_score)

        model_fit.save('./Output/arima_fold_{}_result.dat'.format(fold))

    print('Fold Score List: {}'.format(fold_score_list))

    print()

if __name__ == '__main__':
    main()
