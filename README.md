# DeLELSTM-Decomposition-based-Linear-Explainable-LSTM-to-Capture-Instantaneous-and-Long-term-Effects
## Requirements
Python == 3.8.

Pytorch: 1.8.1+cu102, Sklearn:0.23.2, Numpy: 1.19.2, Pandas: 1.1.3, Matplotlib: 3.3.2

All the codes are run on GPUs by default.

## PM2.5 Experiment
The PM2.5 data can be downloaded from https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

## Electricity Experiment
The electricity data can be downloaded from https://www.kaggle.com/datasets/unajtheb/homesteadus-electricity-consumption

## Exchange Experiment 
The exchange data can be downloaded from https://github.com/laiguokun/multivariate-time-series-data

## Training 
The following commands will train three task-specific datasets. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

```
python3 electricity_experiment.py --model_name decompose_Explain_LSTM_pertime
python3 exchange_experiment.py --model_name decompose_Explain_LSTM_pertime
python3 PM_experiment.py --model_name decompose_Explain_LSTM_pertime
```
