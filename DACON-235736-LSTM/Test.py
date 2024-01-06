import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Util.CustomDataset import CustomDataset
from Util.Mapper import get_model_by_name
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

    # Define Test Data
    last_train_window = pd.read_csv('./Output/last_train_window.csv')
    test_x = np.vstack((last_train_window.values, df.values))

    # Define Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define Dataset, Dataloader
    test_dataset = CustomDataset(test_x, None, Config.window_size)
    
    if Config.fixed_randomness:
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
    else:
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False)

    # Define Modellist, Print Modellist
    model_list = []
    for idx, test_input_model in enumerate(Config.test_input_model_list):
        idx += 1
        
        ckpt = torch.load('./Output/{}'.format(test_input_model))

        test_model_name = ckpt['name']

        print()
        print('Model: {}, Name: {}'.format(idx, test_model_name))

        model = get_model_by_name(test_model_name)
        if Config.print_model:
            print()
            print(model)
        print()
        print('Epoch: {}, Val Loss: {:.4}, Val Score: {:.4}'.format(ckpt['epoch'], ckpt['loss'], ckpt['score']))

        model.load_state_dict(ckpt['model_state_dict'])
        model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        model_list.append(model)

    # Test
    test_pred_list = []

    print()
    with torch.no_grad():
        for input in tqdm(iter(test_loader)):
            input = input.to(device)

            output_sum = torch.zeros((input.size(0), Config.output_size)).to(device)
            for model in model_list:
                output = model(input)
                output_sum = torch.add(output_sum, output)

            pred = torch.div(output_sum, len(model_list))

            test_pred_list.extend(pred.detach().cpu().numpy().tolist())

    print()

    submit = pd.read_csv('./Data/sample_submission.csv')
    if Config.test_run:
        submit = submit.head(Config.test_run_data_size)

    submit['answer'] = test_pred_list
    submit.to_csv('./Output/submit.csv', index=False)

if __name__ == '__main__':
    main()
