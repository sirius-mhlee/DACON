import numpy as np
import pandas as pd

import statsmodels.tsa.arima.model as smt

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Util.Preprocessing import date_preprocessing

def main():
    # Data Load
    df = pd.read_csv('./Data/sample_submission.csv')
    if Config.test_run:
        #df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    print()
    print(df.info())

    # Data Preprocessing
    date_preprocessing(df)

    print()
    print(df.info())

    # Define Modellist, Print Modellist
    model_list = []
    for idx, test_input_model in enumerate(Config.test_input_model_list):
        idx += 1

        print()
        print('Model: {}, Name: {}'.format(idx, test_input_model))

        model = smt.ARIMAResults.load('./Output/{}'.format(test_input_model))
        model_list.append(model)

    # Test
    test_pred_list = np.zeros(len(df))

    for model in model_list:
        test_pred_list += model.predict(start=df.index.min(), end=df.index.max()).values

    test_pred_list /= len(model_list)

    print()

    submit = pd.read_csv('./Data/sample_submission.csv')
    if Config.test_run:
        submit = submit.head(Config.test_run_data_size)

    submit['평균기온'] = test_pred_list
    submit.to_csv('./Output/submit.csv', index=False)

if __name__ == '__main__':
    main()
