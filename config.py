
def config_elec(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim'] = args.input_dim
    params['n_units'] = args.n_units
    params['bias'] = True
    params['time_depth']=args.depth
    params['output_dim']=args.output_dim

    if model_name in ['Delelstm']:
        params['N_units'] = args.N_units

    elif model_name == 'IMV_full':
            params['n_units']=16

    elif model_name == 'IMV_tensor':
             pass

    elif model_name == 'Retain':
        params['intputDimSize'] = args.input_dim
        params['embDimSize'] = 256
        params['alphaHiddenDimSize']=128
        params['betaHiddenDimSize'] = 128
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0

    elif model_name == 'LSTM':
        pass
    else:
        raise ModuleNotFoundError

    return params

def config_pm(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim'] = args.input_dim
    params['n_units'] = args.n_units
    params['bias'] = True
    params['time_depth']=args.depth
    params['output_dim']=args.output_dim

    if model_name in ['Delelstm']:
        params['N_units'] = args.N_units

    elif model_name == 'IMV_full':
            params['n_units']=16

    elif model_name == 'IMV_tensor':
             pass

    elif model_name == 'Retain':
        params['intputDimSize'] = args.input_dim
        params['embDimSize'] =64
        params['alphaHiddenDimSize']=64
        params['betaHiddenDimSize'] = 64
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0
        params['batch_size']=args.batch_size

    elif model_name == 'LSTM':
        pass
    else:
        raise ModuleNotFoundError

    return params

def config_exchange(model_name, args):
    params = {}
    # some common params for all models
    params['input_dim'] = args.input_dim
    params['n_units'] = args.n_units
    params['bias'] = True
    params['time_depth']=args.depth
    params['output_dim']=args.output_dim

    if model_name in ['Delelstm']:
        params['N_units'] = args.N_units

    elif model_name == 'IMV_full':
            params['n_units']=16

    elif model_name == 'IMV_tensor':
             pass

    elif model_name == 'Retain':
        params['intputDimSize'] = args.input_dim
        params['embDimSize'] = 64
        params['alphaHiddenDimSize']=64
        params['betaHiddenDimSize'] = 64
        params['outputDimSize']=args.output_dim
        params['keep_prob']=1.0
        params['batch_size'] = args.batch_size

    elif model_name == 'LSTM':
        pass
    else:
        raise ModuleNotFoundError

    return params


def config(model_name, args):
    if args.dataset == 'electricity':
        params = config_elec(model_name, args)

    elif args.dataset == 'PM':
        params = config_pm(model_name, args)

    elif args.dataset == 'exchange':
        params = config_exchange(model_name, args)

    return params

