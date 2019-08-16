import itertools
import numpy as np
import re

arg_support_enabled = True

def vprint(verbose, string):
    """
    Print string if verbose.

    Prints string if verbose is a True value
    """
    if verbose:
        print(string)

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

def soft_cast_int(n):
    """
    Convert to int, if possible.

    If it is empty, returns None.
    If it is not numeric, it is left unchanged.
    """
    if n == None or n == '':
        return None
    try:
        return int(n)
    except ValueError:
        return n

def soft_cast_float(n):
    """
    Convert to float, if possible.

    If it is empty, returns None.
    If it is not numeric, it is left unchanged.
    """
    if n == None or n == '':
        return None
    try:
        return float(n)
    except ValueError:
        return n


def flatten(somelist):
    """
    Flatten a list.

    Example: flatten([[1, 2], [3], [4, 5]]) -> [1, 2, 3, 4, 5]
    """
    return [item for sublist in somelist for item in sublist]

def parse_conf_str(conf_str):
    conf_dict = dict()
    for option in conf_str.split(','):
        key, value = option.split('=')
        conf_dict[key] = soft_cast_float(value)
    return conf_dict

def remove_index_from_tuple(parameters, index):
    return (*parameters[:index], *parameters[index+1:])

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

def prune_dependent_parameters(by_name, parameter_names):
    """
    Remove dependent parameters from aggregate.

    :param by_name: measurements partitioned by state/transition/... name and attribute, edited in-place.
        by_name[name][attribute] must be a list or 1-D numpy array.
        by_name[stanamete_or_trans]['param'] must be a list of parameter values.
        Other dict members are left as-is
    :param parameter_names: List of parameter names in the order they are used in by_name[name]['param'], edited in-place.

    Model generation (and its components, such as relevant parameter detection and least squares optimization) only work if input variables (i.e., parameters)
    are independent of each other. This function computes the correlation coefficient for each pair of parameters and removes those which depend on each other.
    For each pair of dependent parameters, the lexically greater one is removed (e.g. "a" and "b" -> "b" is removed).
    """

    parameter_indices_to_remove = list()
    for parameter_combination in itertools.product(range(len(parameter_names)), range(len(parameter_names))):
        index_1, index_2 = parameter_combination
        if index_1 >= index_2:
            continue
        parameter_values = [list(), list()] # both parameters have a value
        parameter_values_1 = list() # parameter 1 has a value
        parameter_values_2 = list() # parameter 2 has a value
        for name in by_name:
            for measurement in by_name[name]['param']:
                value_1 = measurement[index_1]
                value_2 = measurement[index_2]
                if is_numeric(value_1):
                    parameter_values_1.append(value_1)
                if is_numeric(value_2):
                    parameter_values_2.append(value_2)
                if is_numeric(value_1) and is_numeric(value_2):
                    parameter_values[0].append(value_1)
                    parameter_values[1].append(value_2)
        if len(parameter_values[0]):
            correlation = np.corrcoef(parameter_values)[0][1]
            if correlation != np.nan and np.abs(correlation) > 0.5:
                print('[!] Parameters {} <-> {} are correlated with coefficcient {}'.format(parameter_names[index_1], parameter_names[index_2], correlation))
                if len(parameter_values_1) < len(parameter_values_2):
                    index_to_remove = index_1
                else:
                    index_to_remove = index_2
                print('    Removing parameter {}'.format(parameter_names[index_to_remove]))
                parameter_indices_to_remove.append(index_to_remove)
    remove_parameters_by_indices(by_name, parameter_names, parameter_indices_to_remove)

def remove_parameters_by_indices(by_name, parameter_names, parameter_indices_to_remove):
    """
    Remove parameters listed in `parameter_indices` from aggregate `by_name` and `parameter_names`.

    :param by_name: measurements partitioned by state/transition/... name and attribute, edited in-place.
        by_name[name][attribute] must be a list or 1-D numpy array.
        by_name[stanamete_or_trans]['param'] must be a list of parameter values.
        Other dict members are left as-is
    :param parameter_names: List of parameter names in the order they are used in by_name[name]['param'], edited in-place.
    :param parameter_indices_to_remove: List of parameter indices to be removed
    """

    # Start removal from the end of the list to avoid renumbering of list elemenets
    for parameter_index in sorted(parameter_indices_to_remove, reverse = True):
        for name in by_name:
            for measurement in by_name[name]['param']:
                measurement.pop(parameter_index)
        parameter_names.pop(parameter_index)

def compute_param_statistics(by_name, by_param, parameter_names, arg_count, state_or_trans, attribute, verbose = False):
    """
    Compute standard deviation and correlation coefficient for various data partitions.

    It is strongly recommended to vary all parameter values evenly across partitions.
    For instance, given two parameters, providing only the combinations
    (1, 1), (5, 1), (7, 1,) (10, 1), (1, 2), (1, 6) will lead to bogus results.
    It is better to provide (1, 1), (5, 1), (1, 2), (5, 2), ... (i.e. a cross product of all individual parameter values)

    :param by_name: ground truth partitioned by state/transition name.
        by_name[state_or_trans][attribute] must be a list or 1-D numpy array.
        by_name[state_or_trans]['param'] must be a list of parameter values
        corresponding to the ground truth, e.g. [[1, 2, 3], ...] if the
        first ground truth element has the (lexically) first parameter set to 1,
        the second to 2 and the third to 3.
    :param by_param: ground truth partitioned by state/transition name and parameters.
        by_name[(state_or_trans, *)][attribute] must be a list or 1-D numpy array.
    :param parameter_names: list of parameter names, must have the same order as the parameter
        values in by_param (lexical sorting is recommended).
    :param arg_count: dict providing the number of functions args ("local parameters") for each function.
    :param state_or_trans: state or transition name, e.g. 'send' or 'TX'
    :param attribute: model attribute, e.g. 'power' or 'duration'
    :param verbose: print warning if some parameter partitions are too small for fitting

    :return: a dict with the following content:
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
        ret['std_by_param'][param] = _mean_std_by_param(by_param, state_or_trans, attribute, param_idx, verbose)
        ret['corr_by_param'][param] = _corr_by_param(by_name, state_or_trans, attribute, param_idx)
    if arg_support_enabled and state_or_trans in arg_count:
        for arg_index in range(arg_count[state_or_trans]):
            ret['std_by_arg'].append(_mean_std_by_param(by_param, state_or_trans, attribute, len(parameter_names) + arg_index, verbose))
            ret['corr_by_arg'].append(_corr_by_param(by_name, state_or_trans, attribute, len(parameter_names) + arg_index))

    return ret

def _mean_std_by_param(by_param, state_or_tran, attribute, param_index, verbose = False):
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
            vprint(verbose, '[W] parameter value partition for {} contains only one element -- skipping'.format(param_value))
        else:
            vprint(verbose, '[W] parameter value partition for {} is empty'.format(param_value))
    if len(partitions) == 0:
        vprint(verbose, '[W] Found no partitions for {}/{}/{} ???'.format(state_or_tran, attribute, param_index))
        return 0.
    return np.mean([np.std(partition) for partition in partitions])

def _corr_by_param(by_name, state_or_trans, attribute, param_index):
    if _all_params_are_numeric(by_name[state_or_trans], param_index):
        param_values = np.array(list((map(lambda x: x[param_index], by_name[state_or_trans]['param']))))
        try:
            return np.corrcoef(by_name[state_or_trans][attribute], param_values)[0, 1]
        except FloatingPointError:
            # Typically happens when all parameter values are identical.
            # Building a correlation coefficient is pointless in this case
            # -> assume no correlation
            return 0.
        except ValueError:
            print('[!] Exception in _corr_by_param(by_name, state_or_trans={}, attribute={}, param_index={})'.format(state_or_trans, attribute, param_index))
            print('[!] while executing np.corrcoef(by_name[{}][{}]={}, {}))'.format(state_or_trans, attribute, by_name[state_or_trans][attribute], param_values))
            raise
    else:
        return 0.

def _all_params_are_numeric(data, param_idx):
    param_values = list(map(lambda x: x[param_idx], data['param']))
    if len(list(filter(is_numeric, param_values))) == len(param_values):
        return True
    return False

class OptionalTimingAnalysis:
    def __init__(self, enabled = True):
        self.enabled = enabled
        self.wrapped_lines = list()
        self.index = 1

    def get_header(self):
        ret = ''
        if self.enabled:
            ret += '#define TIMEIT(index, functioncall) '
            ret += 'counter.start(); '
            ret += 'functioncall; '
            ret += 'counter.stop();'
            ret += 'kout << endl << index << " :: " << counter.value << "/" << counter.overflow << endl;\n'
        return ret

    def wrap_codeblock(self, codeblock):
        if not self.enabled:
            return codeblock
        lines = codeblock.split('\n')
        ret = list()
        for line in lines:
            if re.fullmatch('.+;', line):
                ret.append('TIMEIT( {:d}, {} )'.format(self.index, line))
                self.wrapped_lines.append(line)
                self.index += 1
            else:
                ret.append(line)
        return '\n'.join(ret)
