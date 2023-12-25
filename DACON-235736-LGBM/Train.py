import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

import lightgbm as lgb

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Util.Preprocessing import date_preprocessing, export_preprocessing
from Util.Metric import smape

def main():
    # Data Load
    df = pd.read_csv('./Data/train.csv', encoding='euc-kr')
    if Config.test_run:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    print()
    print(df.info())

    # Data Preprocessing
    date_preprocessing(df)
    export_preprocessing(df)

    print()
    print(df.info())

    # Define Train Data
    train_x = df.drop('전력사용량(kWh)', axis=1)
    train_y = df[['전력사용량(kWh)']]

    # Define KFold
    if Config.use_fold:
        if Config.fixed_randomness:
            kf = KFold(n_splits=Config.fold_k, shuffle=True, random_state=Config.seed)
        else:
            kf = KFold(n_splits=Config.fold_k, shuffle=True)
    else:
        if Config.fixed_randomness:
            split_train_idx, split_val_idx = train_test_split(np.arange(len(train_x)), test_size=0.2, shuffle=True, random_state=Config.seed)
        else:
            split_train_idx, split_val_idx = train_test_split(np.arange(len(train_x)), test_size=0.2, shuffle=True)

    # Define Model Param
    model_params = {
        'objective': 'regression',
        'metric': 'None',
        'learning_rate': Config.learning_rate,
        'verbosity': -1,
    }

    # Train
    fold_score_list = []

    if Config.use_fold:
        data_generator = kf.split(train_x, train_y)
    else:
        data_generator = [(split_train_idx, split_val_idx)]

    print()
    for fold, (train_idx, val_idx) in enumerate(data_generator):
        fold += 1

        print('Fold: {}'.format(fold))

        fold_train_x = train_x.iloc[train_idx, :]
        fold_train_y = train_y.iloc[train_idx, :]

        fold_val_x = train_x.iloc[val_idx, :]
        fold_val_y = train_y.iloc[val_idx, :]

        print()

        data_train = lgb.Dataset(fold_train_x, fold_train_y)
        data_val = lgb.Dataset(fold_val_x, fold_val_y)

        model = lgb.train(params=model_params,
                    train_set=data_train,
                    num_boost_round=Config.num_boost_round,
                    valid_sets=data_val,
                    valid_names='Fold Valid Data',
                    feval=smape,
                    callbacks=[lgb.log_evaluation(period=Config.num_boost_round)])

        print()
        
        fold_pred = model.predict(fold_val_x)
        _, fold_score, _ = smape(fold_pred, data_val)

        fold_score_list.append(fold_score)

        model.save_model('./Output/lgbm_fold_{}_result.dat'.format(fold))

    print('Fold Score List: {}'.format(fold_score_list))

    print()

if __name__ == '__main__':
    main()
