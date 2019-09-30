import itertools
import numpy as np
import re

arg_support_enabled = True

def vprint(verbose, string):
    """
    Print `string` if `verbose`.

    Prints string if verbose is a True value
    """
    if verbose:
        print(string)

def is_numeric(n):
    """Check if `n` is numeric (i.e., it can be converted to float)."""
    if n == None:
        return False
    try:
        float(n)
        return True
    except ValueError:
        return False

def float_or_nan(n):
    """Convert `n` to float (if numeric) or NaN."""
    if n == None:
        return np.nan
    try:
        return float(n)
    except ValueError:
        return np.nan

def soft_cast_int(n):
    """
    Convert `n` to int, if possible.

    If `n` is empty, returns None.
    If `n` is not numeric, it is left unchanged.
    """
    if n == None or n == '':
        return None
    try:
        return int(n)
    except ValueError:
        return n

def soft_cast_float(n):
    """
    Convert `n` to float, if possible.

    If `n` is empty, returns None.
    If `n` is not numeric, it is left unchanged.
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
    """
    Parse a configuration string `k1=v1,k2=v2`... and return a dict `{'k1': v1, 'k2': v2}`...

    Values are casted to float if possible and kept as-is otherwise.
    """
    conf_dict = dict()
    for option in conf_str.split(','):
        key, value = option.split('=')
        conf_dict[key] = soft_cast_float(value)
    return conf_dict

def remove_index_from_tuple(parameters, index):
    """
    Remove the element at `index` from tuple `parameters`.

    :param parameters: tuple
    :param index: index of element which is to be removed
    :returns: parameters tuple without the element at index
    """
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

def prune_dependent_parameters(by_name, parameter_names, correlation_threshold = 0.5):
    """
    Remove dependent parameters from aggregate.

    :param by_name: measurements partitioned by state/transition/... name and attribute, edited in-place.
        by_name[name][attribute] must be a list or 1-D numpy array.
        by_name[stanamete_or_trans]['param'] must be a list of parameter values.
        Other dict members are left as-is
    :param parameter_names: List of parameter names in the order they are used in by_name[name]['param'], edited in-place.
    :param correlation_threshold: Remove parameter if absolute correlation exceeds this threshold (default: 0.5)

    Model generation (and its components, such as relevant parameter detection and least squares optimization) only works if input variables (i.e., parameters)
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
            # Calculating the correlation coefficient only makes sense when neither value is constant
            if np.std(parameter_values_1) != 0 and np.std(parameter_values_2) != 0:
                correlation = np.corrcoef(parameter_values)[0][1]
                if correlation != np.nan and np.abs(correlation) > correlation_threshold:
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

    :returns: a dict with the following content:
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
        std_matrix, mean_std = _std_by_param(by_param, state_or_trans, attribute, param_idx, verbose)
        ret['std_by_param'][param] = mean_std
        ret['corr_by_param'][param] = _corr_by_param(by_name, state_or_trans, attribute, param_idx)
    if arg_support_enabled and state_or_trans in arg_count:
        for arg_index in range(arg_count[state_or_trans]):
            std_matrix, mean_std = _std_by_param(by_param, state_or_trans, attribute, len(parameter_names) + arg_index, verbose)
            ret['std_by_arg'].append(mean_std)
            ret['corr_by_arg'].append(_corr_by_param(by_name, state_or_trans, attribute, len(parameter_names) + arg_index))

    return ret

def _param_values(by_param, state_or_tran):
    """
    Return the distinct values of each parameter in by_param.

    E.g. if by_param.keys() contains the distinct parameter values (1, 1), (1, 2), (1, 3), (0, 3),
    this function returns [[1, 0], [1, 2, 3]].
    Note that the order is not deterministic at the moment.

    Also note that this function deliberately also consider None
    (uninitialized parameter with unknown value) as a distinct value. Benchmarks
    and drivers must ensure that a parameter is only None when its value is
    not important yet, e.g. a packet length parameter must only be None when
    write() or similar has not been called yet. Other parameters should always
    be initialized when leaving UNINITIALIZED.

    """
    param_tuples = list(map(lambda x: x[1], filter(lambda x: x[0] == state_or_tran, by_param.keys())))
    distinct_values = [set() for i in range(len(param_tuples[0]))]
    for param_tuple  in param_tuples:
        for i in range(len(param_tuple)):
            distinct_values[i].add(param_tuple[i])

    # TODO returned values must have a deterministic order

    # Convert sets to lists
    distinct_values = list(map(list, distinct_values))
    return distinct_values

def _std_by_param(by_param, state_or_tran, attribute, param_index, verbose = False):
    u"""
    Calculate standard deviations for a static model where all parameters but param_index are constant.

    :param by_param: measurements sorted by key/transition name and parameter values
    :param state_or_tran: state or transition name (-> by_param[(state_or_tran, *)])
    :param attribute: model attribute, e.g. 'power' or 'duration'
           (-> by_param[(state_or_tran, *)][attribute])
    :param param_index: index of variable parameter
    :returns: (stddev matrix, mean stddev)

    Returns the mean standard deviation of all measurements of 'attribute'
    (e.g. power consumption or timeout) for state/transition 'state_or_tran' where
    parameter 'param_index' is dynamic and all other parameters are fixed.
    I.e., if parameters are a, b, c ∈ {1,2,3} and 'index' corresponds to b, then
    this function returns the mean of the standard deviations of (a=1, b=*, c=1),
    (a=1, b=*, c=2), and so on.
    Also returns an (n-1)-dimensional array (where n is the number of parameters)
    giving the standard deviation of each individual partition. E.g. for
    param_index == 2 and 4 parameters, array[a][b][d] is the
    stddev of measurements with param0 == a, param1 == b, param2 variable,
    and param3 == d.
    """
    # TODO precalculate or cache info_shape (it only depends on state_or_tran)
    param_values = list(remove_index_from_tuple(_param_values(by_param, state_or_tran), param_index))
    info_shape = tuple(map(len, param_values))

    # We will calculate the mean over the entire matrix later on. We cannot
    # guarantee that each entry will be filled in this loop (e.g. transitions
    # whose arguments are combined using 'zip' rather than 'cartesian' always
    # have missing parameter combinations), we pre-fill it with NaN and use
    # np.nanmean to skip those when calculating the mean.
    stddev_matrix = np.full(info_shape, np.nan)

    for param_value in itertools.product(*param_values):
        param_partition = list()
        for k, v in by_param.items():
            if k[0] == state_or_tran and (*k[1][:param_index], *k[1][param_index+1:]) == param_value:
                param_partition.extend(v[attribute])

        if len(param_partition) > 1:
            matrix_index = list(range(len(param_value)))
            for i in range(len(param_value)):
                matrix_index[i] = param_values[i].index(param_value[i])
            matrix_index = tuple(matrix_index)
            stddev_matrix[matrix_index] = np.std(param_partition)
        # This can (and will) happen in normal operation, e.g. when a transition's
        # arguments are combined using 'zip' rather than 'cartesian'.
        #elif len(param_partition) == 1:
        #    vprint(verbose, '[W] parameter value partition for {} contains only one element -- skipping'.format(param_value))
        #else:
        #    vprint(verbose, '[W] parameter value partition for {} is empty'.format(param_value))

    if np.all(np.isnan(stddev_matrix)):
        vprint(verbose, '[W] {}/{} parameter #{} has no data partitions -- how did this even happen?'.format(state_or_tran, attribute, param_index))
        vprint(verbose, 'stddev_matrix = {}'.format(stddev_matrix))
        return stddev_matrix, 0.

    return stddev_matrix, np.nanmean(stddev_matrix) #np.mean([np.std(partition) for partition in partitions])

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
    """Check if all `data['param'][*][param_idx]` elements are numeric, as reported by `utils.is_numeric`."""
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
