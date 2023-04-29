import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torchmetrics import MeanAbsolutePercentageError
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def elec_training(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=20, verbose=True, min_lr=0.0001)

    epochs = 350
    min_val_loss = 1000000
    loss=nn.MSELoss()

    save_path = os.path.join(args.save_dirs, 'Electricity', 'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()


    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        mse_train=0
        j=0
        for batch_x, batch_y in train_loader:
                j += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device) #shape batch, depth
                opt.zero_grad()

                y_pred, unorm_weight,_= model(batch_x.float(), device)
                y_pred = y_pred.squeeze(-1)

                l = loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item()
                opt.step()
        time_elapsed = time.time() - epoch_start_time
        print( f"Training loss at epoch {i}: mse={mse_train/j:.4f}, rmse={(mse_train / (j)) ** 0.5:.4f}, "
               f"time elapsed: {time_elapsed:.2f}")
        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train MSE'] = mse_train / (j)
            df_log_val.loc[i, 'Train RMSE'] = (mse_train / (j)) ** 0.5


        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha=[]
            val_beta=[]
            valweight_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output, unorm_weight,_ = model(batch_x.float(),device)

                output = output.squeeze(-1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                if model_name in ['IMV_full','IMV_tensor']:
                    val_alpha.append(unorm_weight.detach().cpu().numpy())
                    val_beta.append(_.detach().cpu().numpy())
                if model_name in ['Delelstm']:
                    unorm = unorm_weight.squeeze(-1)  # shape time depth, batch, feature
                    valweight_list += [unorm]
                if model_name in ['Retain']:
                    valweight_list += [unorm_weight]
        if model_name in ['Delelstm']:
            valweight = torch.stack(valweight_list)
            valweight = valweight.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim * 2)
            valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight = pd.DataFrame(valweight)
        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            valweight = torch.stack(valweight_list)
            valweight = valweight.reshape(-1, args.depth - 2, args.input_dim)
            valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight = pd.DataFrame(valweight)
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        mse_val = mean_squared_error(true, preds)
        mae_val = mean_absolute_error(true, preds)


        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            print("Saving...")
            model_save_path = os.path.join(save_path, model_name)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path,str(exp_id)+'_elec_predict.pt'))

            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path,str(exp_id) + '_valbeta.csv'))
            if model_name=='Delelstm':
                valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))
                soft_valweight = valweight.copy()

                for row in range(len(soft_valweight)):
                    soft_valweight.loc[row, :] = soft_valweight.loc[row, :].abs() / sum(soft_valweight.loc[row, :].abs())

                soft_valweight.to_csv(os.path.join(model_save_path,str(exp_id) + '_soft_valweight.csv'))

            if model_name == 'Retain':
                valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

        if (i % 10 == 0):
            print("lr: ", opt.param_groups[0]["lr"])
        if args.log:
            df_log_val.loc[i, 'mse_val'] = mse_val
            df_log_val.loc[i, 'rmse_val'] = (mse_val) ** 0.5
            df_log_val.loc[i, 'mae_val'] = mae_val

        epoch_scheduler.step(mse_val)


    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path,str(exp_id)+'_elec_predict.pt')))
        preds = []
        true = []
        test_alpha=[]
        test_beta=[]
        unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output, unorm_weight,_ = model(batch_x.float(), device)  # unorm time, batch, features
            output = output.squeeze(-1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_weight.detach().cpu().numpy())
                test_beta.append(_.detach().cpu().numpy())
            if model_name in ['Delelstm']:
                unorm = unorm_weight.squeeze(-1)  # shape time depth, batch, feature
                unorm_list += [unorm]
            if model_name in ['Retain']:
                # shape time depth, batch, feature
                unorm_list += [unorm_weight]


    preds = np.concatenate(preds)
    true = np.concatenate(true)
    newpred=torch.tensor(preds)
    newtrue=torch.tensor(true)
    if model_name in ['Retain']:
        test_weight = torch.stack(unorm_list)  # shape 19, 5, 100, 5
        test_weight = test_weight.reshape(-1, args.depth - 2, args.input_dim)
        test_weight = test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        test_weight = pd.DataFrame(test_weight)
        test_weight.to_csv(os.path.join(model_save_path,str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['Delelstm']:
        unorm_weight = torch.stack(unorm_list)  # shape 19, 5, 100, 5
        unorm_weight = unorm_weight.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim * 2)
        unorm_weight = unorm_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        unorm_weight = pd.DataFrame(unorm_weight)
        unorm_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
        soft_testweight = unorm_weight.copy()
        for i in range(len(soft_testweight)):
            soft_testweight.loc[i, :] = soft_testweight.loc[i, :].abs() / sum(soft_testweight.loc[i, :].abs())

        soft_testweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight.csv'))

    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path,str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path,str(exp_id) + '_testbeta.csv'))

    mse_test = mean_squared_error(true, preds)
    mae_test = mean_absolute_error(true, preds)
    mean_abs_percentage_error=MeanAbsolutePercentageError()
    mape_test=mean_abs_percentage_error(newpred,newtrue)
    df_log_test.loc[0, 'mse_test'] = mse_test
    df_log_test.loc[0, 'rmse_test'] = (mse_test)** 0.5
    df_log_test.loc[0, 'mae_test'] = mae_test
    df_log_test.loc[0, 'mape_test'] = mape_test.item()

    print('mse, rmse, mae,mape', mse_test, mse_test ** 0.5, mae_test,mape_test.item())

    outcome = np.hstack((true, preds))
    outcome = pd.DataFrame(outcome)
    #
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path,str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))


def exchange_training(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    #opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=20, verbose=True, min_lr=0.0001)

    epochs = 300
    min_val_loss = 1000000
    loss=nn.MSELoss()

    save_path = os.path.join(args.save_dirs, 'Exchange', 'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()


    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        mse_train=0
        j=0
        for batch_x, batch_y in train_loader:
                j += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device) #shape batch, depth
                opt.zero_grad()

                y_pred, unorm_weight,_= model(batch_x.float(), device)
                y_pred = y_pred.squeeze(-1)

                l = loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item()
                opt.step()
        time_elapsed = time.time() - epoch_start_time
        print( f"Training loss at epoch {i}: mse={mse_train/j:.4f}, rmse={(mse_train / (j)) ** 0.5:.4f}, "
               f"time elapsed: {time_elapsed:.2f}")
        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train MSE'] = mse_train / (j)
            df_log_val.loc[i, 'Train RMSE'] = (mse_train / (j)) ** 0.5


        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha=[]
            val_beta=[]
            valweight_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output, unorm_weight,_ = model(batch_x.float(),device)

                output = output.squeeze(-1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                if model_name in ['IMV_full','IMV_tensor']:
                    val_alpha.append(unorm_weight.detach().cpu().numpy())
                    val_beta.append(_.detach().cpu().numpy())
                if model_name in ['Delelstm']:
                    unorm = unorm_weight.squeeze(-1)  # shape time depth, batch, feature
                    valweight_list += [unorm]
                if model_name in ['Retain']:
                    valweight_list += [unorm_weight]
        if model_name in ['Delelstm']:
            valweight = torch.stack(valweight_list)
            valweight = valweight.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim * 2)
            valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight = pd.DataFrame(valweight)
        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            valweight = torch.stack(valweight_list)
            valweight = valweight.reshape(-1, args.depth - 2, args.input_dim)
            valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight = pd.DataFrame(valweight)
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        mse_val = mean_squared_error(true, preds)
        mae_val = mean_absolute_error(true, preds)


        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            print("Saving...")
            model_save_path = os.path.join(save_path, model_name)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path,str(exp_id)+'_exchange_predict.pt'))

            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path,str(exp_id) + '_valbeta.csv'))
            if model_name=='Delelstm':
                valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))
                soft_valweight = valweight.copy()

                for row in range(len(soft_valweight)):
                    soft_valweight.loc[row, :] = soft_valweight.loc[row, :].abs() / sum(soft_valweight.loc[row, :].abs())

                soft_valweight.to_csv(os.path.join(model_save_path,str(exp_id) + '_soft_valweight.csv'))

            if model_name == 'Retain':
                valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

        if (i % 10 == 0):
            print("lr: ", opt.param_groups[0]["lr"])
        if args.log:
            df_log_val.loc[i, 'mse_val'] = mse_val
            df_log_val.loc[i, 'rmse_val'] = (mse_val) ** 0.5
            df_log_val.loc[i, 'mae_val'] = mae_val

        epoch_scheduler.step(mse_val)


    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path,str(exp_id)+'_exchange_predict.pt')))
        preds = []
        true = []
        test_alpha=[]
        test_beta=[]
        unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output, unorm_weight,_ = model(batch_x.float(), device)  # unorm time, batch, features
            output = output.squeeze(-1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_weight.detach().cpu().numpy())
                test_beta.append(_.detach().cpu().numpy())
            if model_name in ['Delelstm']:
                unorm = unorm_weight.squeeze(-1)  # shape time depth, batch, feature
                unorm_list += [unorm]
            if model_name in ['Retain']:
                # shape time depth, batch, feature
                unorm_list += [unorm_weight]


    preds = np.concatenate(preds)
    true = np.concatenate(true)
    newpred=torch.tensor(preds)
    newtrue=torch.tensor(true)
    if model_name in ['Retain']:
        test_weight = torch.stack(unorm_list)  # shape 19, 5, 100, 5
        test_weight = test_weight.reshape(-1, args.depth - 2, args.input_dim)
        test_weight = test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        test_weight = pd.DataFrame(test_weight)
        test_weight.to_csv(os.path.join(model_save_path,str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['Delelstm']:
        unorm_weight = torch.stack(unorm_list)  # shape 19, 5, 100, 5
        unorm_weight = unorm_weight.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim * 2)
        unorm_weight = unorm_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        unorm_weight = pd.DataFrame(unorm_weight)
        unorm_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
        soft_testweight = unorm_weight.copy()
        for i in range(len(soft_testweight)):
            soft_testweight.loc[i, :] = soft_testweight.loc[i, :].abs() / sum(soft_testweight.loc[i, :].abs())

        soft_testweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight.csv'))

    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path,str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path,str(exp_id) + '_testbeta.csv'))

    mse_test = mean_squared_error(true, preds)
    mae_test = mean_absolute_error(true, preds)
    mean_abs_percentage_error=MeanAbsolutePercentageError()
    mape_test=mean_abs_percentage_error(newpred,newtrue)
    df_log_test.loc[0, 'mse_test'] = mse_test
    df_log_test.loc[0, 'rmse_test'] = (mse_test)** 0.5
    df_log_test.loc[0, 'mae_test'] = mae_test
    df_log_test.loc[0, 'mape_test'] = mape_test.item()

    print('mse, rmse, mae,mape', mse_test, mse_test ** 0.5, mae_test,mape_test.item())

    outcome = np.hstack((true, preds))
    outcome = pd.DataFrame(outcome)
    #
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path,str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))

def PM_training(model, model_name, train_loader, val_loader, test_loader, args, device, exp_id):
    #opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.03)
    opt = torch.optim.Adam(model.parameters(), lr=0.05,weight_decay=0.03)
    epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=20, verbose=True, min_lr=0.0001)

    #epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=32)
    epochs =300
    min_val_loss = 1000000
    loss=nn.MSELoss()

    save_path = os.path.join(args.save_dirs, 'newrevision_pmc', 'exp_' + str(exp_id))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()
        df_log_test = pd.DataFrame()


    print(f'Experiment: {exp_id}')
    for i in range(epochs):
        '''Model training'''
        epoch_start_time = time.time()
        model.train()
        mse_train=0
        j=0
        for batch_x, batch_y in train_loader:
                j += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device) #shape batch, depth
                opt.zero_grad()

                y_pred, unorm_weight,_= model(batch_x.float(), device)
                y_pred = y_pred.squeeze(-1)

                l = loss(y_pred, batch_y)
                l.backward()
                mse_train += l.item()
                opt.step()
        time_elapsed = time.time() - epoch_start_time
        print( f"Training loss at epoch {i}: mse={mse_train/j:.4f}, rmse={(mse_train / (j)) ** 0.5:.4f}, "
               f"time elapsed: {time_elapsed:.2f}")
        if args.log:
            df_log_val.loc[i, 'Epoch'] = i
            df_log_val.loc[i, 'Train MSE'] = mse_train / (j)
            df_log_val.loc[i, 'Train RMSE'] = (mse_train / (j)) ** 0.5


        '''Model validation'''
        with torch.no_grad():
            model.eval()
            preds = []
            true = []
            val_alpha=[]
            val_beta=[]
            valweight_list = torch.jit.annotate(list[Tensor], [])
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output, unorm_weight,_ = model(batch_x.float(),device)

                output = output.squeeze(-1)
                preds.append(output.detach().cpu().numpy())
                true.append(batch_y.detach().cpu().numpy())
                if model_name in ['IMV_full','IMV_tensor']:
                    val_alpha.append(unorm_weight.detach().cpu().numpy())
                    val_beta.append(_.detach().cpu().numpy())
                if model_name in ['Delelstm']:
                    unorm = unorm_weight.squeeze(-1)  # shape time depth, batch, feature
                    valweight_list += [unorm]
                if model_name in ['Retain']:
                    valweight_list += [unorm_weight]
        if model_name in ['Delelstm']:
            valweight = torch.stack(valweight_list)
            valweight = valweight.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim * 2)
            valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight = pd.DataFrame(valweight)
        if model_name in ['IMV_full', 'IMV_tensor']:
            val_alpha = np.concatenate(val_alpha)
            val_beta = np.concatenate(val_beta)
            val_alpha = val_alpha.mean(axis=0)
            val_beta = val_beta.mean(axis=0)
            val_beta = pd.DataFrame(val_beta)
            val_alpha = pd.DataFrame(val_alpha)
        if model_name in ['Retain']:
            valweight = torch.stack(valweight_list)
            valweight = valweight.reshape(-1, args.depth - 2, args.input_dim)
            valweight = valweight.mean(dim=0, keepdim=False).detach().cpu().numpy()
            valweight = pd.DataFrame(valweight)
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        mse_val = mean_squared_error(true, preds)
        mae_val = mean_absolute_error(true, preds)


        if min_val_loss > mse_val ** 0.5:
            min_val_loss = mse_val ** 0.5
            print("Saving...")
            model_save_path = os.path.join(save_path, model_name)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path,str(exp_id)+'_pm_predict.pt'))

            if model_name in ['IMV_full', 'IMV_tensor']:
                val_alpha.to_csv(os.path.join(model_save_path, str(exp_id) + '_valalpha.csv'))
                val_beta.to_csv(os.path.join(model_save_path,str(exp_id) + '_valbeta.csv'))
            if model_name=='Delelstm':
                valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))
                soft_valweight = valweight.copy()

                for row in range(len(soft_valweight)):
                    soft_valweight.loc[row, :] = soft_valweight.loc[row, :].abs() / sum(soft_valweight.loc[row, :].abs())

                soft_valweight.to_csv(os.path.join(model_save_path,str(exp_id) + '_soft_valweight.csv'))
            if model_name == 'Retain':
                valweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_valweight.csv'))

        if (i % 10 == 0):
            print("lr: ", opt.param_groups[0]["lr"])
        if args.log:
            df_log_val.loc[i, 'mse_val'] = mse_val
            df_log_val.loc[i, 'rmse_val'] = (mse_val) ** 0.5
            df_log_val.loc[i, 'mae_val'] = mae_val
        #epoch_scheduler.step()
        epoch_scheduler.step(mse_val)


    '''Modeling test'''
    print(f'Testing:')
    with torch.no_grad():
        model.eval()
        # load the best model parameter
        model.load_state_dict(torch.load(os.path.join(model_save_path,str(exp_id)+'_pm_predict.pt')))
        preds = []
        true = []
        test_alpha=[]
        test_beta=[]
        unorm_list = torch.jit.annotate(list[Tensor], [])
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output, unorm_weight,_ = model(batch_x.float(), device)  # unorm time, batch, features
            output = output.squeeze(-1)
            preds.append(output.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            if model_name in ['IMV_full', 'IMV_tensor']:
                test_alpha.append(unorm_weight.detach().cpu().numpy())
                test_beta.append(_.detach().cpu().numpy())
            if model_name in ['Delelstm']:
                unorm = unorm_weight.squeeze(-1)  # shape time depth, batch, feature
                unorm_list += [unorm]
            if model_name in ['Retain']:
                # shape time depth, batch, feature
                unorm_list += [unorm_weight]


    preds = np.concatenate(preds)
    true = np.concatenate(true)
    newpred=torch.tensor(preds)
    newtrue=torch.tensor(true)
    if model_name in ['Retain']:
        test_weight = torch.stack(unorm_list)  # shape 19, 5, 100, 5
        test_weight = test_weight.reshape(-1, args.depth - 2, args.input_dim)
        test_weight = test_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        test_weight = pd.DataFrame(test_weight)
        test_weight.to_csv(os.path.join(model_save_path,str(exp_id) + '_Explain_test_weight.csv'))
    if model_name in ['Delelstm']:
        unorm_weight = torch.stack(unorm_list)  # shape 19, 5, 100, 5
        unorm_weight = unorm_weight.permute(0, 2, 1, 3).reshape(-1, args.depth - 2, args.input_dim * 2)
        unorm_weight = unorm_weight.mean(dim=0, keepdim=False).detach().cpu().numpy()  # shape time, feature
        unorm_weight = pd.DataFrame(unorm_weight)
        unorm_weight.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_weight.csv'))
        soft_testweight = unorm_weight.copy()
        for i in range(len(soft_testweight)):
            soft_testweight.loc[i, :] = soft_testweight.loc[i, :].abs() / sum(soft_testweight.loc[i, :].abs())

        soft_testweight.to_csv(os.path.join(model_save_path, str(exp_id) + '_soft_testweight.csv'))

    if model_name in ['IMV_full', 'IMV_tensor']:
        test_alpha = np.concatenate(test_alpha)
        test_beta = np.concatenate(test_beta)
        test_alpha = test_alpha.mean(axis=0)
        test_beta = test_beta.mean(axis=0)
        test_beta = pd.DataFrame(test_beta)
        test_alpha = pd.DataFrame(test_alpha)
        test_alpha.to_csv(os.path.join(model_save_path,str(exp_id) + '_testalpha.csv'))
        test_beta.to_csv(os.path.join(model_save_path,str(exp_id) + '_testbeta.csv'))

    mse_test = mean_squared_error(true, preds)
    mae_test = mean_absolute_error(true, preds)
    mean_abs_percentage_error=MeanAbsolutePercentageError()
    mape_test=mean_abs_percentage_error(newpred,newtrue)
    df_log_test.loc[0, 'mse_test'] = mse_test
    df_log_test.loc[0, 'rmse_test'] = (mse_test)** 0.5
    df_log_test.loc[0, 'mae_test'] = mae_test
    df_log_test.loc[0, 'mape_test'] = mape_test.item()

    print('mse, rmse, mae,mape', mse_test, mse_test ** 0.5, mae_test,mape_test.item())

    outcome = np.hstack((true, preds))
    outcome = pd.DataFrame(outcome)
    #
    outcome.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_outcome.csv'))
    df_log_val.to_csv(os.path.join(model_save_path,str(exp_id) + '_Expalin_train_results.csv'))
    df_log_test.to_csv(os.path.join(model_save_path, str(exp_id) + '_Explain_test_results.csv'))

