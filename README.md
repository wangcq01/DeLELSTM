# DeLELSTM: Decomposition-based Linear Explainable LSTM to Capture Instantaneous and Long term Effects
## Requirements
Python == 3.9.

Pytorch: 2.0.0+cu117, Numpy: 1.23.5, Pandas: 1.5.3, Matplotlib: 3.7.1

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
python3 Electricity.py --model_name Delelstm
python3 Exchange.py --model_name Delelstm
python3 PM.py --model_name Delelstm
```
