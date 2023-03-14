import pandas as pd

import pickle

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Model.CustomAlexnet import *

from Util.CustomDataset import CustomDataset

def main():
    # Data Load
    df = pd.read_csv('./Data/test.csv')
    if Config.test_run:
        #df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    le = pickle.load(open('./Output/encoder.pkl', 'rb'))

    ckpt = torch.load('./Output/{}'.format(Config.test_input_model))

    test_img_paths = df['img_path'].values

    # Define Device, Print Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print()
    print(CustomAlexnet(class_num=10))

    # Define Transform, Dataset, Dataloader
    test_transform = A.Compose([
                                A.Resize(Config.img_size, Config.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_dataset = CustomDataset(test_img_paths, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=True)

    # Define Model, Criterion, Optimizer, Scheduler  
    model = CustomAlexnet(class_num=10)
    model.load_state_dict(ckpt['model_state_dict'])
    model = nn.DataParallel(model)
    model.to(device)

    print()
    print('Epoch: {}, Val Loss: {:.4}, Val Score: {:.4}'.format(ckpt['epoch'], ckpt['loss'], ckpt['score']))

    # Test
    test_pred_list = []

    print()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for input in tqdm(iter(test_loader)):
            input = input.to(device)
            
            output = model(input)
            _, pred = torch.max(output, 1)

            test_pred_list.extend(pred.detach().cpu().numpy().tolist())

    print()

    submit = pd.read_csv('./Data/sample_submission.csv')
    if Config.test_run:
        submit = submit.head(Config.test_run_data_size)

    submit['target'] = le.inverse_transform(test_pred_list)
    submit.to_csv('./Output/submit.csv', index=False)

if __name__ == '__main__':
    main()
