from itertools import chain, combinations
import numpy as np
import re
from scipy import optimize
from utils import is_numeric

arg_support_enabled = True

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class ParamFunction:

    def __init__(self, param_function, validation_function, num_vars):
        self._param_function = param_function
        self._validation_function = validation_function
        self._num_variables = num_vars

    def is_valid(self, arg):
        return self._validation_function(arg)

    def eval(self, param, args):
        return self._param_function(param, args)

    def error_function(self, P, X, y):
        return self._param_function(P, X) - y

class AnalyticFunction:

    def __init__(self, function_str, parameters, num_args, verbose = True, regression_args = None):
        self._parameter_names = parameters
        self._num_args = num_args
        self._model_str = function_str
        rawfunction = function_str
        self._dependson = [False] * (len(parameters) + num_args)
        self.fit_success = False
        self.verbose = verbose

        if type(function_str) == str:
            num_vars_re = re.compile(r'regression_arg\(([0-9]+)\)')
            num_vars = max(map(int, num_vars_re.findall(function_str))) + 1
            for i in range(len(parameters)):
                if rawfunction.find('parameter({})'.format(parameters[i])) >= 0:
                    self._dependson[i] = True
                    rawfunction = rawfunction.replace('parameter({})'.format(parameters[i]), 'model_param[{:d}]'.format(i))
            for i in range(0, num_args):
                if rawfunction.find('function_arg({:d})'.format(i)) >= 0:
                    self._dependson[len(parameters) + i] = True
                    rawfunction = rawfunction.replace('function_arg({:d})'.format(i), 'model_param[{:d}]'.format(len(parameters) + i))
            for i in range(num_vars):
                rawfunction = rawfunction.replace('regression_arg({:d})'.format(i), 'reg_param[{:d}]'.format(i))
            self._function_str = rawfunction
            self._function = eval('lambda reg_param, model_param: ' + rawfunction)
        else:
            self._function_str = 'raise ValueError'
            self._function = function_str

        if regression_args:
            self._regression_args = regression_args.copy()
            self._fit_success = True
        elif type(function_str) == str:
            self._regression_args = list(np.ones((num_vars)))
        else:
            self._regression_args = []

    def get_fit_data(self, by_param, state_or_tran, model_attribute):
        dimension = len(self._parameter_names) + self._num_args
        X = [[] for i in range(dimension)]
        Y = []

        num_valid = 0
        num_total = 0

        for key, val in by_param.items():
            if key[0] == state_or_tran and len(key[1]) == dimension:
                valid = True
                num_total += 1
                for i in range(dimension):
                    if self._dependson[i] and not is_numeric(key[1][i]):
                        valid = False
                if valid:
                    num_valid += 1
                    Y.extend(val[model_attribute])
                    for i in range(dimension):
                        if self._dependson[i]:
                            X[i].extend([float(key[1][i])] * len(val[model_attribute]))
                        else:
                            X[i].extend([np.nan] * len(val[model_attribute]))
            elif key[0] == state_or_tran and len(key[1]) != dimension:
                vprint(self.verbose, '[W] Invalid parameter key length while gathering fit data for {}/{}. is {}, want {}.'.format(state_or_tran, model_attribute, len(key[1]), dimension))
        X = np.array(X)
        Y = np.array(Y)

        return X, Y, num_valid, num_total

    def fit(self, by_param, state_or_tran, model_attribute):
        X, Y, num_valid, num_total = self.get_fit_data(by_param, state_or_tran, model_attribute)
        if num_valid > 2:
            error_function = lambda P, X, y: self._function(P, X) - y
            try:
                res = optimize.least_squares(error_function, self._regression_args, args=(X, Y), xtol=2e-15)
            except ValueError as err:
                vprint(self.verbose, '[W] Fit failed for {}/{}: {} (function: {})'.format(state_or_tran, model_attribute, err, self._model_str))
                return
            if res.status > 0:
                self._regression_args = res.x
                self.fit_success = True
            else:
                vprint(self.verbose, '[W] Fit failed for {}/{}: {} (function: {})'.format(state_or_tran, model_attribute, res.message, self._model_str))
        else:
            vprint(self.verbose, '[W] Insufficient amount of valid parameter keys, cannot fit {}/{}'.format(state_or_tran, model_attribute))

    def is_predictable(self, param_list):
        for i, param in enumerate(param_list):
            if self._dependson[i] and not is_numeric(param):
                return False
        return True

    def eval(self, param_list, arg_list = []):
        if len(self._regression_args) == 0:
            return self._function(param_list, arg_list)
        return self._function(self._regression_args, param_list)

class analytic:
    _num0_8 = np.vectorize(lambda x: 8 - bin(int(x)).count("1"))
    _num0_16 = np.vectorize(lambda x: 16 - bin(int(x)).count("1"))
    _num1 = np.vectorize(lambda x: bin(int(x)).count("1"))
    _safe_log = np.vectorize(lambda x: np.log(np.abs(x)) if np.abs(x) > 0.001 else 1.)
    _safe_inv = np.vectorize(lambda x: 1 / x if np.abs(x) > 0.001 else 1.)
    _safe_sqrt = np.vectorize(lambda x: np.sqrt(np.abs(x)))

    _function_map = {
        'linear' : lambda x: x,
        'logarithmic' : np.log,
        'logarithmic1' : lambda x: np.log(x + 1),
        'exponential' : np.exp,
        'square' : lambda x : x ** 2,
        'inverse' : lambda x : 1 / x,
        'sqrt' : lambda x: np.sqrt(np.abs(x)),
        'num0_8' : _num0_8,
        'num0_16' : _num0_16,
        'num1' : _num1,
        'safe_log' : lambda x: np.log(np.abs(x)) if np.abs(x) > 0.001 else 1.,
        'safe_inv' : lambda x: 1 / x if np.abs(x) > 0.001 else 1.,
        'safe_sqrt': lambda x: np.sqrt(np.abs(x)),
    }

    def functions(safe_functions_enabled = False):
        functions = {
            'linear' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * model_param,
                lambda model_param: True,
                2
            ),
            'logarithmic' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * np.log(model_param),
                lambda model_param: model_param > 0,
                2
            ),
            'logarithmic1' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * np.log(model_param + 1),
                lambda model_param: model_param > -1,
                2
            ),
            'exponential' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * np.exp(model_param),
                lambda model_param: model_param <= 64,
                2
            ),
            #'polynomial' : lambda reg_param, model_param: reg_param[0] + reg_param[1] * model_param + reg_param[2] * model_param ** 2,
            'square' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * model_param ** 2,
                lambda model_param: True,
                2
            ),
            'inverse' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] / model_param,
                lambda model_param: model_param != 0,
                2
            ),
            'sqrt' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * np.sqrt(model_param),
                lambda model_param: model_param >= 0,
                2
            ),
            'num0_8' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * analytic._num0_8(model_param),
                lambda model_param: True,
                2
            ),
            'num0_16' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * analytic._num0_16(model_param),
                lambda model_param: True,
                2
            ),
            'num1' : ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * analytic._num1(model_param),
                lambda model_param: True,
                2
            ),
        }

        if safe_functions_enabled:
            functions['safe_log'] = ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * analytic._safe_log(model_param),
                lambda model_param: True,
                2
            )
            functions['safe_inv'] = ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * analytic._safe_inv(model_param),
                lambda model_param: True,
                2
            )
            functions['safe_sqrt'] = ParamFunction(
                lambda reg_param, model_param: reg_param[0] + reg_param[1] * analytic._safe_sqrt(model_param),
                lambda model_param: True,
                2
            )

        return functions

    def _fmap(reference_type, reference_name, function_type):
        ref_str = '{}({})'.format(reference_type,reference_name)
        if function_type == 'linear':
            return ref_str
        if function_type == 'logarithmic':
            return 'np.log({})'.format(ref_str)
        if function_type == 'logarithmic1':
            return 'np.log({} + 1)'.format(ref_str)
        if function_type == 'exponential':
            return 'np.exp({})'.format(ref_str)
        if function_type == 'exponential':
            return 'np.exp({})'.format(ref_str)
        if function_type == 'square':
            return '({})**2'.format(ref_str)
        if function_type == 'inverse':
            return '1/({})'.format(ref_str)
        if function_type == 'sqrt':
            return 'np.sqrt({})'.format(ref_str)
        return 'analytic._{}({})'.format(function_type, ref_str)

    def function_powerset(function_descriptions, parameter_names, num_args):
        buf = '0'
        arg_idx = 0
        for combination in powerset(function_descriptions.items()):
            buf += ' + regression_arg({:d})'.format(arg_idx)
            arg_idx += 1
            for function_item in combination:
                if arg_support_enabled and is_numeric(function_item[0]):
                    buf += ' * {}'.format(analytic._fmap('function_arg', function_item[0], function_item[1]['best']))
                else:
                    buf += ' * {}'.format(analytic._fmap('parameter', function_item[0], function_item[1]['best']))
        return AnalyticFunction(buf, parameter_names, num_args)
