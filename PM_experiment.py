import time
import pandas as pd
import numpy as np
from networks import   decompose_Explain_LSTM_pertime,normalLSTMpertime, IMVFullLSTM_pertime,IMVTensorLSTM_pertime,Retain_pertime
import torch
from torch import nn, Tensor
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

start_time=time.time()


data_path=r'C:\clean\Explain_time_series\mts-interpretability-benchmark-main\DATA\PM25'
data = pd.read_csv(os.path.join(data_path,'newX_train.csv'))

y=pd.read_csv(os.path.join(data_path,'newy_train.csv'))
depth = 24
data_idx=np.array(list(set(data['idx'])))

N=len(set(data['idx']))
cols = list(data.columns[6:])
for col in cols:
     data[col] = (data[col] - np.min(data[col])) / (
                 np.max(data[(col)]) - np.min(data[col]))


train_idx = np.random.choice(data_idx, int(0.75 * N), replace=False)
data_idx = data_idx[~np.in1d(data_idx, train_idx)]
val_idx = np.random.choice(data_idx, int(0.15 * N), replace=False)
data_idx = data_idx[~np.in1d(data_idx, val_idx)]
test_idx = data_idx

train_X = data.loc[data['idx'].isin(train_idx), :]
new_train=train_X.iloc[:,6:]
new_train = np.array(new_train)
new_train = torch.tensor(new_train)
train_Y = y.loc[y['idx'].isin(train_idx), :]

val_X = data.loc[data['idx'].isin(val_idx), :]
new_val=val_X.iloc[:,6:]
new_val = np.array(new_val)
new_val = torch.tensor(new_val)
val_Y =y.loc[y['idx'].isin(val_idx), :]

test_X = data.loc[data['idx'].isin(test_idx), :]
new_test=test_X.iloc[:,6:]
new_test = np.array(new_test)
new_test = torch.tensor(new_test)
test_Y = y.loc[y['idx'].isin(test_idx), :]



X_train_t=new_train.reshape(len(train_idx),depth,len(cols))
X_val_t=new_val.reshape(len(val_idx),depth,len(cols))
X_test_t=new_test.reshape(len(test_idx),depth,len(cols))
batch_size=32


y_train_t = torch.Tensor(train_Y['pm2.5'].to_numpy()).reshape(len(train_idx), depth-1)
y_val_t = torch.Tensor(val_Y['pm2.5'].to_numpy()).reshape(len(val_idx), depth-1)
y_test_t = torch.Tensor(test_Y['pm2.5'].to_numpy()).reshape(len(test_idx), depth-1)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False,drop_last=True)



depth=24
input_dim=len(cols)
output_dim=1
n_units=128
#N_units=n_units*input_dim
N_units=128



save_path = os.path.join('newresults', 'PM25/decompose')
if not os.path.exists(save_path):
    os.makedirs(save_path)

for exp_i in range(5):
    model = decompose_Explain_LSTM_pertime(input_dim, output_dim, n_units,N_units, depth).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=32)


    epochs = 1000
    loss = nn.MSELoss()
    patience = 35
    min_val_loss = 9999
    counter = 0
    df_log_val = pd.DataFrame()
    df_log_test = pd.DataFrame()

    for i in range(epochs):
        print('epoch:{}'.format(i))
        count = 0
        mse_train = 0
        j=0
        for batch_x, batch_y in train_loader:
            j +=1
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            opt.zero_grad()

            y_pred, unorm_weight = model(batch_x.float())

            unorm_weight=unorm_weight.squeeze(-1) # shape time depth, batch, feature
            y_pred = y_pred.squeeze(-1)

            l = loss(y_pred, batch_y)
            l.backward()

            mse_train += l.item()
            opt.step()
        epoch_scheduler.step()
        df_log_val.loc[i, 'Epoch'] = i
        df_log_val.loc[i, 'Train MSE'] = mse_train / (j)
        df_log_val.loc[i, 'Train RMSE'] = (mse_train / (j)) ** 0.5

        # val dataset

        with torch.no_grad():

            preds = []
            true = []
            valweight_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                output, unorm_weight = model(batch_x.float())
                unorm = unorm_weight.squeeze(-1) # shape time depth, batch, feature
                valweight_list += [unorm]
                output = output.squeeze(-1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())


        valweight = torch.stack(valweight_list)
        valweight=valweight.permute(0,2,1,3).reshape(-1,depth-1, input_dim*2)
        valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
        valweight = pd.DataFrame(valweight)
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        mse_val = mean_squared_error(true, preds)
        mae_val = mean_absolute_error(true, preds)

        df_log_val.loc[i, 'mse_val'] = mse_val
        df_log_val.loc[i, 'rmse_val'] = (mse_val) ** 0.5
        df_log_val.loc[i, 'mae_val'] = mae_val

        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            print("Saving...")
            torch.save(model.state_dict(), "decompose_pm25.pt")
            valweight.to_csv(os.path.join(save_path, str(exp_i) + 'valweight.csv'))
            soft_valweight = valweight.copy()

            for i in range(len(soft_valweight)):

                soft_valweight.loc[i, :] = soft_valweight.loc[i, :].abs()/ sum(soft_valweight.loc[i, :].abs())

            soft_valweight.to_csv(os.path.join(save_path, str(exp_i) + 'soft_valweight.csv'))
            counter = 0
        else:
            counter += 1

        if counter == patience:
            break
        print("Iter: ", i, "train: ", (mse_train / (j)) ** 0.5, "val: ", (mse_val) ** 0.5)

        if (i % 10 == 0):
            print("lr: ", opt.param_groups[0]["lr"])



    # test data
    model.load_state_dict(torch.load('decompose_pm25.pt'))
    with torch.no_grad():
        preds = []
        true = []
        unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output, unorm = model(batch_x.float())  # unorm time, batch, features
            unorm=unorm.squeeze(-1)
            unorm_list += [unorm]

            output = output.squeeze(-1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds)
    true = np.concatenate(true)
    unorm_weight = torch.stack(unorm_list)
    unorm_weight=unorm_weight.permute(0,2,1,3).reshape(-1,depth-1,input_dim*2)
    unorm_weight = unorm_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
    unorm_weight = pd.DataFrame(unorm_weight)
    unorm_weight.to_csv(os.path.join(save_path, str(exp_i) + 'Explain_test_weight.csv'))
    soft_testweight = unorm_weight.copy()
    for i in range(len(soft_testweight)):
        soft_testweight.loc[i, :] = soft_testweight.loc[i, :].abs() / sum(soft_testweight.loc[i, :].abs())

    soft_testweight.to_csv(os.path.join(save_path, str(exp_i) + 'soft_testweight.csv'))


    mse_test = mean_squared_error(true, preds)
    mae_test = mean_absolute_error(true, preds)
    df_log_test.loc[0, 'mse_test'] = mse_test
    df_log_test.loc[0, 'rmse_test'] = (mse_test)** 0.5
    df_log_test.loc[0, 'mae_test'] = mae_test
    print('mse, rmse, mae', mse_test, mse_test ** 0.5, mae_test)

    outcome = np.hstack((true, preds))
    outcome = pd.DataFrame(outcome)
    #
    outcome.to_csv(os.path.join(save_path, str(exp_i) + 'Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(save_path, str(exp_i) + 'Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(save_path, str(exp_i) + 'Explain_test_results.csv'))

end_time = time.time()
print('time elapse', end_time - start_time)













