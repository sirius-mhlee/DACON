import numpy as np
import pandas as pd

import lightgbm as lgb

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Util.Preprocessing import date_preprocessing, import_preprocessing

def main():
    # Data Load
    df = pd.read_csv('./Data/test.csv', encoding='euc-kr')
    if Config.test_run:
        #df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    print()
    print(df.info())

    # Data Preprocessing
    date_preprocessing(df)
    import_preprocessing(df)

    print()
    print(df.info())

    # Define Modellist, Print Modellist
    model_list = []
    for idx, test_input_model in enumerate(Config.test_input_model_list):
        idx += 1

        print()
        print('Model: {}, Name: {}'.format(idx, test_input_model))

        model = lgb.Booster(model_file='./Output/{}'.format(test_input_model))
        model_list.append(model)

    # Test
    test_pred_list = np.zeros(len(df))

    for model in model_list:
        test_pred_list += model.predict(df)

    test_pred_list /= len(model_list)

    print()

    submit = pd.read_csv('./Data/sample_submission.csv')
    if Config.test_run:
        submit = submit.head(Config.test_run_data_size)

    submit['answer'] = test_pred_list
    submit.to_csv('./Output/submit.csv', index=False)

if __name__ == '__main__':
    main()
