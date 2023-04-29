import os
import argparse
import numpy as np
import pandas as pd
import config
from networks import Delelstm,IMVTensorLSTM_pertime,IMVFullLSTM_pertime,Retain_pertime,normalLSTMpertime
from model_training import elec_training
import torch
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description="Electricity consumption prediction")
parser.add_argument('--seed', type=int, default=555, help='The random seed')
parser.add_argument('--depth', type=int, default=24, help='The length of time series')
parser.add_argument('--input_dim', type=int, default=16, help='The dimension of dataset')
parser.add_argument('--output_dim', type=int, default=1, help='The dimension for output')
parser.add_argument('--N_units', type=int, default=64, help='The hidden size for vanilla LSTM')
parser.add_argument('--n_units', type=int, default=64, help='The hidden size for Tensor LSTM')
parser.add_argument('--dataset', type=str, default='electricity')
parser.add_argument('--save_dirs', type=str, default='results', help='The dirs for saving results')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size when training NN')
parser.add_argument('--num_exp', type=int, default=5, help='The number of experiment')
parser.add_argument('--log', type=bool, default=True, help='Whether log the information of training process')
parser.add_argument('--save_models', type=bool, default=False, help='Whether save the training models')


args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    data_path = r'C:\clean\Explain_time_series\Code\DATA\Processed\elec'
    data = pd.read_csv(os.path.join(data_path, 'newX_train.csv'))

    y = pd.read_csv(os.path.join(data_path, 'second_y.csv'))
    data_idx = np.array(list(set(data['idx'])))

    N = len(set(data['idx']))
    cols = list(data.columns[2:])
    for col in cols:
        data[col] = (data[col] - np.min(data[col])) / (
                np.max(data[(col)]) - np.min(data[col]))

    train_idx = np.random.choice(data_idx, int(0.75 * N), replace=False)
    data_idx = data_idx[~np.in1d(data_idx, train_idx)]
    val_idx = np.random.choice(data_idx, int(0.15 * N), replace=False)
    data_idx = data_idx[~np.in1d(data_idx, val_idx)]
    test_idx = data_idx

    train_X = data.loc[data['idx'].isin(train_idx), :]
    new_train = train_X.iloc[:, 2:]
    new_train = np.array(new_train)
    new_train = torch.tensor(new_train)
    train_Y = y.loc[y['idx'].isin(train_idx), :]

    val_X = data.loc[data['idx'].isin(val_idx), :]
    new_val = val_X.iloc[:, 2:]
    new_val = np.array(new_val)
    new_val = torch.tensor(new_val)
    val_Y = y.loc[y['idx'].isin(val_idx), :]

    test_X = data.loc[data['idx'].isin(test_idx), :]
    new_test = test_X.iloc[:, 2:]
    new_test = np.array(new_test)
    new_test = torch.tensor(new_test)
    test_Y = y.loc[y['idx'].isin(test_idx), :]

    X_train_t = new_train.reshape(len(train_idx), args.depth, len(cols))
    X_val_t = new_val.reshape(len(val_idx), args.depth, len(cols))
    X_test_t = new_test.reshape(len(test_idx), args.depth, len(cols))

    y_train_t = torch.Tensor(train_Y['target'].to_numpy()).reshape(len(train_idx), args.depth - 2)
    y_val_t = torch.Tensor(val_Y['target'].to_numpy()).reshape(len(val_idx), args.depth - 2)
    y_test_t = torch.Tensor(test_Y['target'].to_numpy()).reshape(len(test_idx), args.depth - 2)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=args.batch_size, shuffle=False, drop_last=True)

    save_path = os.path.join(args.save_dirs, 'Electricity')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    short=0
    model_list = ['Delelstm', 'IMV_full', 'IMV_tensor', 'Retain', 'LSTM']
    #model_list = ['LSTM']
    # split the training set and test set and construct the data loader

    for exp_id in range(args.num_exp):
        for model_name in model_list:
            if model_name == 'Delelstm':
                model = Delelstm(config.config(model_name, args), short).to(device)
            elif model_name == 'IMV_full':
                model = IMVFullLSTM_pertime(config.config(model_name, args),short).to(device)

            elif model_name == 'IMV_tensor':
                model = IMVTensorLSTM_pertime(config.config(model_name, args),short).to(device)

            elif model_name == 'Retain':
                model = Retain_pertime(config.config(model_name, args), short).to(device)

            elif model_name == 'LSTM':
                model = normalLSTMpertime(config.config(model_name, args), short).to(device)

            else:
                ModuleNotFoundError(f'Module {model_name} not found')
            print(f'Training model {model_name}')
            print(f'Num of trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
            elec_training(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id)
            print()