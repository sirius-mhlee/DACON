import copy
import numpy as np
import pandas as pd

import pickle

from tqdm.auto import tqdm

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Model.CustomAlexnet import *
from Model.Loss import CustomCrossEntropyLoss

from Util.CustomDataset import CustomDataset
from Util.Metric import macro_f1_score

def main():
    # Data Load
    df = pd.read_csv('./Data/train.csv')
    if Config.test_run:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    le = preprocessing.LabelEncoder()
    df['target'] = le.fit_transform(df['target'].values)
    pickle.dump(le, open('./Output/encoder.pkl', 'wb'))

    df.sort_values(by=['id'])
    img_paths = df['img_path'].values
    labels = df['target'].values

    # Define Device, Print Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print()
    print(CustomAlexnet(class_num=10))
    print()

    # Define KFold, Transform
    if Config.fixed_randomness:
        skf = StratifiedKFold(n_splits=Config.fold_k, shuffle=True, random_state=Config.seed)
    else:
        skf = StratifiedKFold(n_splits=Config.fold_k, shuffle=True)

    train_transform = A.Compose([
                                A.Resize(Config.img_size, Config.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                                A.Resize(Config.img_size, Config.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    fold_best_loss = []
    fold_best_score = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        fold += 1
        print('Fold: {}'.format(fold))

        # Define Dataset, Dataloader
        train_img_paths = img_paths[train_idx]
        train_labels = labels[train_idx]

        val_img_paths = img_paths[val_idx]
        val_labels = labels[val_idx]

        train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
        val_dataset = CustomDataset(val_img_paths, val_labels, val_transform)

        if Config.fixed_randomness:
            train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=True, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
            val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=True, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
        else:
            train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=True)

        # Define Model, Criterion, Optimizer, Scheduler
        model = CustomAlexnet(class_num=10)
        model = nn.DataParallel(model)
        model.to(device)

        criterion = CustomCrossEntropyLoss()
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
        val_label_list = []

        print()
        for epoch in range(Config.epoch):
            model.train()
            train_loss = 0.0    
            for input, label in tqdm(iter(train_loader)):
                input = input.to(device)
                label = label.to(device).long()

                optimizer.zero_grad()
                
                output = model(input)
                _, pred = torch.max(output, 1)

                loss = criterion(output, label)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * input.size(0)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for input, label in tqdm(iter(val_loader)):
                    input = input.to(device)
                    label = label.to(device).long()
                    
                    output = model(input)
                    _, pred = torch.max(output, 1)
                    
                    loss = criterion(output, label)
                    
                    val_loss += loss.item() * input.size(0)

                    val_pred_list.extend(pred.detach().cpu().numpy().tolist())
                    val_label_list.extend(label.detach().cpu().numpy().tolist())
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)

            train_loss_list.append(epoch_train_loss)
            val_loss_list.append(epoch_val_loss)

            val_score = macro_f1_score(val_label_list, val_pred_list)
            
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

        fold_best_loss.append(best_loss)
        fold_best_score.append(best_score)

        torch.save({'epoch': best_epoch,
                    'loss': best_loss,
                    'score': best_score,
                    'model_state_dict': best_model},
                    './Output/fold_{}_{}'.format(fold, Config.train_output_model))

    print('Fold Best Loss: {}'.format(fold_best_loss))
    print('Fold Best Score: {}'.format(fold_best_score))
    print()

if __name__ == '__main__':
    main()
