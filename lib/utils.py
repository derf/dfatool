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

def is_power_of_two(n):
    """Check if `n` is a power of two (1, 2, 4, 8, 16, ...)."""
    return n > 0 and (n & (n-1)) == 0

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
    Convert `n` to int (if numeric) or return it as-is.

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
    Convert `n` to float (if numeric) or return it as-is.

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
