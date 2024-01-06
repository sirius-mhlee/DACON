import copy
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Model.Loss import CustomMSELoss

from Util.CustomDataset import CustomDataset
from Util.Mapper import get_model_by_name
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
    train_x = df.drop('전력사용량(kWh)', axis=1).values
    train_y = df[['전력사용량(kWh)']].values

    # Save Last Train Window
    pd.DataFrame(train_x[-Config.window_size:]).to_csv('./Output/last_train_window.csv', index=False)

    # Define Device, Print Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define Train Valid
    split_train_idx, split_val_idx = train_test_split(np.arange(len(train_x)), test_size=0.2, shuffle=False)

    for idx, train_model_name in enumerate(Config.train_model_name_list):
        idx += 1

        print()
        print('Model: {}, Name: {}'.format(idx, train_model_name))
        if Config.print_model:
            print()
            print(get_model_by_name(train_model_name))
        print()

        fold_best_epoch = []
        fold_best_loss = []
        fold_best_score = []

        data_generator = [(split_train_idx, split_val_idx)]

        for fold, (train_idx, val_idx) in enumerate(data_generator):
            fold += 1

            print('Fold: {}'.format(fold))

            # Define Dataset, Dataloader
            fold_train_x = train_x[train_idx]
            fold_train_y = train_y[train_idx]

            fold_val_x = train_x[val_idx]
            fold_val_y = train_y[val_idx]

            train_dataset = CustomDataset(fold_train_x, fold_train_y, Config.window_size)
            val_dataset = CustomDataset(fold_val_x, fold_val_y, Config.window_size)

            if Config.fixed_randomness:
                train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
                val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
            else:
                train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False)
                val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False)

            # Define Model, Criterion, Optimizer, Scheduler
            model = get_model_by_name(train_model_name)
            model = nn.DataParallel(model)
            model.to(device)

            criterion = CustomMSELoss()
            optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

            # Train
            train_loss_list = []
            val_loss_list = []

            best_epoch = 0
            best_loss = np.inf
            best_score = 0.0
            best_model = copy.deepcopy(model.module.state_dict())

            val_pred_list = []
            val_target_list = []

            print()
            for epoch in range(Config.epoch):
                epoch += 1

                model.train()
                train_loss = 0.0    
                for input, target in tqdm(iter(train_loader)):
                    input = input.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()
                    
                    output = model(input)
                    loss = criterion(output, target)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * input.size(0)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for input, target in tqdm(iter(val_loader)):
                        input = input.to(device)
                        target = target.to(device)
                        
                        output = model(input)
                        loss = criterion(output, target)

                        val_loss += loss.item() * input.size(0)

                        val_pred_list.extend(output.detach().cpu().numpy().tolist())
                        val_target_list.extend(target.detach().cpu().numpy().tolist())
                
                epoch_train_loss = train_loss / len(train_loader)
                epoch_val_loss = val_loss / len(val_loader)

                train_loss_list.append(epoch_train_loss)
                val_loss_list.append(epoch_val_loss)

                val_score = smape(val_target_list, val_pred_list)
                
                if scheduler is not None:
                    epoch_lr = scheduler.get_last_lr()[0]
                    scheduler.step()
                else:
                    epoch_lr = Config.learning_rate

                print('Epoch: {}, Learning Rate: {:.6}, Train Loss: {:.4}, Val Loss: {:.4}, Val Score: {:.4}'.format(epoch, epoch_lr, epoch_train_loss, epoch_val_loss, val_score))
                print()

                if epoch_val_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_val_loss
                    best_score = val_score
                    best_model = copy.deepcopy(model.module.state_dict())

            fold_best_epoch.append(best_epoch)
            fold_best_loss.append(best_loss)
            fold_best_score.append(best_score)

            torch.save({'epoch': best_epoch,
                        'loss': best_loss,
                        'score': best_score,
                        'name': train_model_name,
                        'model_state_dict': best_model},
                        './Output/{}_fold_{}_result.pt'.format(train_model_name, fold))

        print('Fold Best Epoch: {}'.format(fold_best_epoch))
        print('Fold Best Loss: {}'.format(fold_best_loss))
        print('Fold Best Score: {}'.format(fold_best_score))

    print()

if __name__ == '__main__':
    main()
