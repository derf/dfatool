import numpy as np

arg_support_enabled = True

def is_numeric(n):
    if n == None:
        return False
    try:
        int(n)
        return True
    except ValueError:
        return False

def param_slice_eq(a, b, index):
    """
    Check if by_param keys a and b are identical, ignoring the parameter at index.

    parameters:
    a, b -- (state/transition name, [parameter0 value, parameter1 value, ...])
    index -- parameter index to ignore (0 -> parameter0, 1 -> parameter1, etc.)

    Returns True iff a and b have the same state/transition name, and all
    parameters at positions != index are identical.

    example:
    ('foo', [1, 4]), ('foo', [2, 4]), 0 -> True
    ('foo', [1, 4]), ('foo', [2, 4]), 1 -> False
    """
    if (*a[1][:index], *a[1][index+1:]) == (*b[1][:index], *b[1][index+1:]) and a[0] == b[0]:
        return True
    return False

def compute_param_statistics(by_name, by_param, parameter_names, num_args, state_or_trans, key):
    ret = {
        'std_static' : np.std(by_name[state_or_trans][key]),
        'std_param_lut' : np.mean([np.std(by_param[x][key]) for x in by_param.keys() if x[0] == state_or_trans]),
        'std_by_param' : {},
        'std_by_arg' : [],
        'corr_by_param' : {},
        'corr_by_arg' : [],
    }

    for param_idx, param in enumerate(parameter_names):
        ret['std_by_param'][param] = _mean_std_by_param(by_param, state_or_trans, key, param_idx)
        ret['corr_by_param'][param] = _corr_by_param(by_name, state_or_trans, key, param_idx)
    if arg_support_enabled and state_or_trans in num_args:
        for arg_index in range(num_args[state_or_trans]):
            ret['std_by_arg'].append(_mean_std_by_param(by_param, state_or_trans, key, len(parameter_names) + arg_index))
            ret['corr_by_arg'].append(_corr_by_param(by_name, state_or_trans, key, len(parameter_names) + arg_index))

    return ret

def _mean_std_by_param(by_param, state_or_tran, key, param_index):
    u"""
    Calculate the mean standard deviation for a static model where all parameters but param_index are constant.

    arguments:
    by_param -- measurements sorted by key/transition name and parameter values
    state_or_tran -- state or transition name (-> by_param[(state_or_tran, *)])
    key -- model attribute, e.g. 'power' or 'duration'
           (-> by_param[(state_or_tran, *)][key])
    param_index -- index of variable parameter

    Returns the mean standard deviation of all measurements of 'key'
    (e.g. power consumption or timeout) for state/transition 'state_or_tran' where
    parameter 'param_index' is dynamic and all other parameters are fixed.
    I.e., if parameters are a, b, c ∈ {1,2,3} and 'index' corresponds to b, then
    this function returns the mean of the standard deviations of (a=1, b=*, c=1),
    (a=1, b=*, c=2), and so on.
    """
    partitions = []
    for param_value in filter(lambda x: x[0] == state_or_tran, by_param.keys()):
        param_partition = []
        for k, v in by_param.items():
            if param_slice_eq(k, param_value, param_index):
                param_partition.extend(v[key])
        if len(param_partition):
            partitions.append(param_partition)
        else:
            print('[W] parameter value partition for {} is empty'.format(param_value))
    return np.mean([np.std(partition) for partition in partitions])

def _corr_by_param(by_name, state_or_trans, key, param_index):
    if _all_params_are_numeric(by_name[state_or_trans], param_index):
        param_values = np.array(list((map(lambda x: x[param_index], by_name[state_or_trans]['param']))))
        try:
            return np.corrcoef(by_name[state_or_trans][key], param_values)[0, 1]
        except FloatingPointError as fpe:
            # Typically happens when all parameter values are identical.
            # Building a correlation coefficient is pointless in this case
            # -> assume no correlation
            return 0.
    else:
        return 0.

def _all_params_are_numeric(data, param_idx):
    param_values = list(map(lambda x: x[param_idx], data['param']))
    if len(list(filter(is_numeric, param_values))) == len(param_values):
        return True
    return False
