import numpy as np

arg_support_enabled = True

def is_numeric(n):
    """Check if n is numeric (i.e., can be converted to int)."""
    if n == None:
        return False
    try:
        int(n)
        return True
    except ValueError:
        return False

def float_or_nan(n):
    """Convert to float (if numeric) or NaN."""
    if n == None:
        return np.nan
    try:
        return float(n)
    except ValueError:
        return np.nan

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

def compute_param_statistics(by_name, by_param, parameter_names, arg_count, state_or_trans, attribute):
    """
    Compute standard deviation and correlation coefficient for various data partitions.

    It is strongly recommended to vary all parameter values evenly across partitions.
    For instance, given two parameters, providing only the combinations
    (1, 1), (5, 1), (7, 1,) (10, 1), (1, 2), (1, 6) will lead to bogus results.
    It is better to provide (1, 1), (5, 1), (1, 2), (5, 2), ... (i.e. a cross product of all individual parameter values)

    arguments:
    by_name -- ground truth partitioned by state/transition name.
        by_name[state_or_trans][attribute] must be a list or 1-D numpy array.
        by_name[state_or_trans]['param'] must be a list of parameter values
        corresponding to the ground truth, e.g. [[1, 2, 3], ...] if the
        first ground truth element has the (lexically) first parameter set to 1,
        the second to 2 and the third to 3.
    by_param -- ground truth partitioned by state/transition name and parameters.
        by_name[(state_or_trans, *)][attribute] must be a list or 1-D numpy array.
    parameter_names -- list of parameter names, must have the same order as the parameter
        values in by_param (lexical sorting is recommended).
    arg_count -- dict providing the number of functions args ("local parameters") for each function.
    state_or_trans -- state or transition name, e.g. 'send' or 'TX'
    attribute -- model attribute, e.g. 'power' or 'duration'

    returns a dict with the following content:
    std_static -- static parameter-unaware model error: stddev of by_name[state_or_trans][attribute]
    std_param_lut -- static parameter-aware model error: mean stddev of by_param[(state_or_trans, *)][attribute]
    std_by_param -- static parameter-aware model error ignoring a single parameter.
        dictionary with one key per parameter. The value is the mean stddev
        of measurements where all other parameters are fixed and the parameter
        in question is variable. E.g. std_by_param['X'] is the mean stddev of
        by_param[(state_or_trans, (X=*, Y=..., Z=...))][attribute].
    std_by_arg -- same, but ignoring a single function argument
        Only set if state_or_trans appears in arg_count, empty dict otherwise.
    corr_by_param -- correlation coefficient
    corr_by_arg -- same, but ignoring a single function argument
        Only set if state_or_trans appears in arg_count, empty dict otherwise.
    """
    ret = {
        'std_static' : np.std(by_name[state_or_trans][attribute]),
        'std_param_lut' : np.mean([np.std(by_param[x][attribute]) for x in by_param.keys() if x[0] == state_or_trans]),
        'std_by_param' : {},
        'std_by_arg' : [],
        'corr_by_param' : {},
        'corr_by_arg' : [],
    }

    np.seterr('raise')

    for param_idx, param in enumerate(parameter_names):
        ret['std_by_param'][param] = _mean_std_by_param(by_param, state_or_trans, attribute, param_idx)
        ret['corr_by_param'][param] = _corr_by_param(by_name, state_or_trans, attribute, param_idx)
    if arg_support_enabled and state_or_trans in arg_count:
        for arg_index in range(arg_count[state_or_trans]):
            ret['std_by_arg'].append(_mean_std_by_param(by_param, state_or_trans, attribute, len(parameter_names) + arg_index))
            ret['corr_by_arg'].append(_corr_by_param(by_name, state_or_trans, attribute, len(parameter_names) + arg_index))

    return ret

def _mean_std_by_param(by_param, state_or_tran, attribute, param_index):
    u"""
    Calculate the mean standard deviation for a static model where all parameters but param_index are constant.

    arguments:
    by_param -- measurements sorted by key/transition name and parameter values
    state_or_tran -- state or transition name (-> by_param[(state_or_tran, *)])
    attribute -- model attribute, e.g. 'power' or 'duration'
           (-> by_param[(state_or_tran, *)][attribute])
    param_index -- index of variable parameter

    Returns the mean standard deviation of all measurements of 'attribute'
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
                param_partition.extend(v[attribute])
        if len(param_partition) > 1:
            partitions.append(param_partition)
        elif len(param_partition) == 1:
            print('[W] parameter value partition for {} contains only one element -- skipping'.format(param_value))
        else:
            print('[W] parameter value partition for {} is empty'.format(param_value))
    return np.mean([np.std(partition) for partition in partitions])

def _corr_by_param(by_name, state_or_trans, attribute, param_index):
    if _all_params_are_numeric(by_name[state_or_trans], param_index):
        param_values = np.array(list((map(lambda x: x[param_index], by_name[state_or_trans]['param']))))
        try:
            return np.corrcoef(by_name[state_or_trans][attribute], param_values)[0, 1]
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
