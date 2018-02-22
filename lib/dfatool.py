#!/usr/bin/env python3

import csv
from itertools import chain, combinations
import io
import json
import numpy as np
import os
from scipy import optimize
from scipy.cluster.vq import kmeans2
import struct
import sys
import tarfile
from multiprocessing import Pool

arg_support_enabled = True

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def is_numeric(n):
    if n == None:
        return False
    try:
        int(n)
        return True
    except ValueError:
        return False

def _soft_cast_int(n):
    if n == None or n == '':
        return None
    try:
        return int(n)
    except ValueError:
        return n

def float_or_nan(n):
    if n == None:
        return np.nan
    try:
        return float(n)
    except ValueError:
        return np.nan

def _elem_param_and_arg_list(elem):
    param_dict = elem['parameter']
    paramkeys = sorted(param_dict.keys())
    paramvalue = [_soft_cast_int(param_dict[x]) for x in paramkeys]
    if arg_support_enabled and 'args' in elem:
        paramvalue.extend(map(_soft_cast_int, elem['args']))
    return paramvalue

def _arg_name(arg_index):
    return '~arg{:02}'.format(arg_index)

def append_if_set(aggregate, data, key):
    if key in data:
        aggregate.append(data[key])

def mean_or_none(arr):
    if len(arr):
        return np.mean(arr)
    return -1

def aggregate_measures(aggregate, actual):
    aggregate_array = np.array([aggregate] * len(actual))
    return regression_measures(aggregate_array, np.array(actual))

def regression_measures(predicted, actual):
    if type(predicted) != np.ndarray:
        raise ValueError('first arg must be ndarray, is {}'.format(type(predicted)))
    if type(actual) != np.ndarray:
        raise ValueError('second arg must be ndarray, is {}'.format(type(actual)))
    deviations = predicted - actual
    if len(deviations) == 0:
        return {}
    measures = {
        'mae' : np.mean(np.abs(deviations), dtype=np.float64),
        'msd' : np.mean(deviations**2, dtype=np.float64),
        'rmsd' : np.sqrt(np.mean(deviations**2), dtype=np.float64),
        'ssr' : np.sum(deviations**2, dtype=np.float64),
    }

    if np.all(actual != 0):
        measures['mape'] = np.mean(np.abs(deviations / actual)) * 100 # bad measure
    if np.all(np.abs(predicted) + np.abs(actual) != 0):
        measures['smape'] = np.mean(np.abs(deviations) / (( np.abs(predicted) + np.abs(actual)) / 2 )) * 100

    return measures

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class Keysight:

    def __init__(self):
        pass

    def load_data(self, filename):
        with open(filename) as f:
            for i, l in enumerate(f):
                pass
            timestamps = np.ndarray((i-3), dtype=float)
            currents = np.ndarray((i-3), dtype=float)
        # basically seek back to start
        with open(filename) as f:
            for _ in range(4):
                next(f)
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                timestamps[i] = float(row[0])
                currents[i] = float(row[2]) * -1
        return timestamps, currents

def _preprocess_measurement(measurement):
    setup = measurement['setup']
    mim = MIMOSA(float(setup['mimosa_voltage']), int(setup['mimosa_shunt']))
    charges, triggers = mim.load_data(measurement['content'])
    trigidx = mim.trigger_edges(triggers)
    triggers = []
    cal_edges = mim.calibration_edges(running_mean(mim.currents_nocal(charges[0:trigidx[0]]), 10))
    calfunc, caldata = mim.calibration_function(charges, cal_edges)
    vcalfunc = np.vectorize(calfunc, otypes=[np.float64])

    processed_data = {
        'fileno' : measurement['fileno'],
        'info' : measurement['info'],
        'triggers' : len(trigidx),
        'first_trig' : trigidx[0] * 10,
        'calibration' : caldata,
        'trace' : mim.analyze_states(charges, trigidx, vcalfunc)
    }

    return processed_data

class RawData:

    def __init__(self, filenames):
        self.filenames = filenames.copy()
        self.traces_by_fileno = []
        self.setup_by_fileno = []
        self.version = 0
        self.preprocessed = False
        self._parameter_names = None

    def _state_is_too_short(self, online, offline, state_duration, next_transition):
        # We cannot control when an interrupt causes a state to be left
        if next_transition['plan']['level'] == 'epilogue':
            return False

        # Note: state_duration is stored as ms, not us
        return offline['us'] < state_duration * 500

    def _state_is_too_long(self, online, offline, state_duration, prev_transition):
        # If the previous state was left by an interrupt, we may have some
        # waiting time left over. So it's okay if the current state is longer
        # than expected.
        if prev_transition['plan']['level'] == 'epilogue':
            return False
        # state_duration is stored as ms, not us
        return offline['us'] > state_duration * 1500

    def _measurement_is_valid(self, processed_data):
        setup = self.setup_by_fileno[processed_data['fileno']]
        traces = self.traces_by_fileno[processed_data['fileno']]
        state_duration = setup['state_duration']
        # Check trigger count
        sched_trigger_count = 0
        for run in traces:
            sched_trigger_count += len(run['trace'])
        if sched_trigger_count != processed_data['triggers']:
            processed_data['error'] = 'got {got:d} trigger edges, expected {exp:d}'.format(
                    got = processed_data['triggers'],
                    exp = sched_trigger_count
            )
            return False
        # Check state durations. Very short or long states can indicate a
        # missed trigger signal which wasn't detected due to duplicate
        # triggers elsewhere
        online_datapoints = []
        for run_idx, run in enumerate(traces):
            for trace_part_idx in range(len(run['trace'])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, online_ref in enumerate(online_datapoints):
            online_run_idx, online_trace_part_idx = online_ref
            offline_trace_part = processed_data['trace'][offline_idx]
            online_trace_part = traces[online_run_idx]['trace'][online_trace_part_idx]

            if self._parameter_names == None:
                self._parameter_names = sorted(online_trace_part['parameter'].keys())

            if sorted(online_trace_part['parameter'].keys()) != self._parameter_names:
                processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) has inconsistent paramete set: should be {param_want:s}, is {param_is:s}'.format(
                    off_idx = offline_idx, on_idx = online_run_idx,
                    on_sub = online_trace_part_idx,
                    on_name = online_trace_part['name'],
                    param_want = self._parameter_names,
                    param_is = sorted(online_trace_part['parameter'].keys())
                )

            if online_trace_part['isa'] != offline_trace_part['isa']:
                processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) claims to be {off_isa:s}, but should be {on_isa:s}'.format(
                        off_idx = offline_idx, on_idx = online_run_idx,
                        on_sub = online_trace_part_idx,
                        on_name = online_trace_part['name'],
                        off_isa = offline_trace_part['isa'],
                        on_isa = online_trace_part['isa'])
                return False

            # Clipping in UNINITIALIZED (offline_idx == 0) can happen during
            # calibration and is handled by MIMOSA
            if offline_idx != 0 and offline_trace_part['clip_rate'] != 0:
                processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) was clipping {clip:f}% of the time'.format(
                    off_idx = offline_idx, on_idx = online_run_idx,
                    on_sub = online_trace_part_idx,
                    on_name = online_trace_part['name'],
                    clip = offline_trace_part['clip_rate'] * 100,
                )
                return False


            if online_trace_part['isa'] == 'state' and online_trace_part['name'] != 'UNINITIALIZED':
                online_prev_transition = traces[online_run_idx]['trace'][online_trace_part_idx-1]
                online_next_transition = traces[online_run_idx]['trace'][online_trace_part_idx+1]
                try:
                    if self._state_is_too_short(online_trace_part, offline_trace_part, state_duration, online_next_transition):
                        processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) is too short (duration = {dur:d} us)'.format(
                            off_idx = offline_idx, on_idx = online_run_idx,
                            on_sub = online_trace_part_idx,
                            on_name = online_trace_part['name'],
                            dur = offline_trace_part['us'])
                        return False
                    if self._state_is_too_long(online_trace_part, offline_trace_part, state_duration, online_prev_transition):
                        processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) is too long (duration = {dur:d} us)'.format(
                            off_idx = offline_idx, on_idx = online_run_idx,
                            on_sub = online_trace_part_idx,
                            on_name = online_trace_part['name'],
                            dur = offline_trace_part['us'])
                        return False
                except KeyError:
                    pass
                    # TODO es gibt next_transitions ohne 'plan'
        return True

    def _merge_measurement_into_online_data(self, measurement):
        online_datapoints = []
        traces = self.traces_by_fileno[measurement['fileno']]
        for run_idx, run in enumerate(traces):
            for trace_part_idx in range(len(run['trace'])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, online_ref in enumerate(online_datapoints):
            online_run_idx, online_trace_part_idx = online_ref
            offline_trace_part = measurement['trace'][offline_idx]
            online_trace_part = traces[online_run_idx]['trace'][online_trace_part_idx]

            if not 'offline' in online_trace_part:
                online_trace_part['offline'] = [offline_trace_part]
            else:
                online_trace_part['offline'].append(offline_trace_part)

            paramkeys = sorted(online_trace_part['parameter'].keys())
            paramvalue = [_soft_cast_int(online_trace_part['parameter'][x]) for x in paramkeys]

            # NB: Unscheduled transitions do not have an 'args' field set.
            # However, they should only be caused by interrupts, and
            # interrupts don't have args anyways.
            if arg_support_enabled and 'args' in online_trace_part:
                paramvalue.extend(map(_soft_cast_int, online_trace_part['args']))

            if not 'offline_aggregates' in online_trace_part:
                online_trace_part['offline_aggregates'] = {
                    'power' : [],
                    'duration' : [],
                    'power_std' : [],
                    'energy' : [],
                    'paramkeys' : [],
                    'param': [],
                }
                if online_trace_part['isa'] == 'transition':
                    online_trace_part['offline_aggregates']['timeout'] = []
                    online_trace_part['offline_aggregates']['rel_energy_prev'] = []
                    online_trace_part['offline_aggregates']['rel_energy_next'] = []

            # Note: All state/transitions are 20us "too long" due to injected
            # active wait states. These are needed to work around MIMOSA's
            # relatively low sample rate of 100 kHz (10us) and removed here.
            online_trace_part['offline_aggregates']['power'].append(
                offline_trace_part['uW_mean'])
            online_trace_part['offline_aggregates']['duration'].append(
                offline_trace_part['us'] - 20)
            online_trace_part['offline_aggregates']['power_std'].append(
                offline_trace_part['uW_std'])
            online_trace_part['offline_aggregates']['energy'].append(
                offline_trace_part['uW_mean'] * (offline_trace_part['us'] - 20))
            online_trace_part['offline_aggregates']['paramkeys'].append(paramkeys)
            online_trace_part['offline_aggregates']['param'].append(paramvalue)
            if online_trace_part['isa'] == 'transition':
                online_trace_part['offline_aggregates']['timeout'].append(
                    offline_trace_part['timeout'])
                online_trace_part['offline_aggregates']['rel_energy_prev'].append(
                    offline_trace_part['uW_mean_delta_prev'] * (offline_trace_part['us'] - 20))
                online_trace_part['offline_aggregates']['rel_energy_next'].append(
                    offline_trace_part['uW_mean_delta_next'] * (offline_trace_part['us'] - 20))

    def _concatenate_analyzed_traces(self):
        self.traces = []
        for trace in self.traces_by_fileno:
            self.traces.extend(trace)

    def get_preprocessed_data(self, verbose = True):
        self.verbose = verbose
        if self.preprocessed:
            return self.traces
        if self.version == 0:
            self.preprocess_0()
        self.preprocessed = True
        return self.traces

    # Loads raw MIMOSA data and turns it into measurements which are ready to
    # be analyzed.
    def preprocess_0(self):
        mim_files = []
        for i, filename in enumerate(self.filenames):
            with tarfile.open(filename) as tf:
                self.setup_by_fileno.append(json.load(tf.extractfile('setup.json')))
                self.traces_by_fileno.append(json.load(tf.extractfile('src/apps/DriverEval/DriverLog.json')))
                for member in tf.getmembers():
                    _, extension = os.path.splitext(member.name)
                    if extension == '.mim':
                        mim_files.append({
                            'content' : tf.extractfile(member).read(),
                            'fileno' : i,
                            'info' : member,
                            'setup' : self.setup_by_fileno[i],
                            'traces' : self.traces_by_fileno[i],
                        })
        with Pool() as pool:
            measurements = pool.map(_preprocess_measurement, mim_files)

        num_valid = 0
        for measurement in measurements:
            if self._measurement_is_valid(measurement):
                self._merge_measurement_into_online_data(measurement)
                num_valid += 1
            elif self.verbose:
                print('[W] Skipping {ar:s}/{m:s}: {e:s}'.format(
                    ar = self.filenames[measurement['fileno']],
                    m = measurement['info'].name,
                    e = measurement['error']))
        if self.verbose:
            print('[I] {num_valid:d}/{num_total:d} measurements are valid'.format(
                num_valid = num_valid,
                num_total = len(measurements)))
        self._concatenate_analyzed_traces()
        self.preprocessing_stats = {
            'num_runs' : len(measurements),
            'num_valid' : num_valid
        }

def _param_slice_eq(a, b, index):
    if (*a[1][:index], *a[1][index+1:]) == (*b[1][:index], *b[1][index+1:]) and a[0] == b[0]:
        return True
    return False

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

    def __init__(self, function_str, num_vars, parameters, num_args):
        self._parameter_names = parameters
        self._num_args = num_args
        self._model_str = function_str
        rawfunction = function_str
        self._dependson = [False] * (len(parameters) + num_args)

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
        self._function = eval('lambda reg_param, model_param: ' + rawfunction);
        self._regression_args = list(np.ones((num_vars)))

    def _get_fit_data(self, by_param, state_or_tran, model_attribute):
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
                print('[W] Invalid parameter key length while gathering fit data for {}/{}. is {}, want {}.'.format(state_or_tran, model_attribute, len(key[1]), dimension))
        for i in range(dimension):
            X[i] = np.array(X[i])
        Y = np.array(Y)

        return X, Y, num_valid, num_total

    def fit(self, by_param, state_or_tran, model_attribute):
        X, Y, num_valid, num_total = self._get_fit_data(by_param, state_or_tran, model_attribute)
        if state_or_tran == 'send':
            print('{} {} dependson {}'.format(state_or_tran, model_attribute, self._dependson))
        if num_valid > 2:
            error_function = lambda P, X, y: self._function(P, X) - y
            try:
                res = optimize.least_squares(error_function, self._regression_args, args=(X, Y), xtol=2e-15)
            except ValueError as err:
                print('[W] Fit failed for {}/{}: {} (function: {})'.format(state_or_tran, model_attribute, err, self._model_str))
                return
            if res.status > 0:
                self._regression_args = res.x
            else:
                print('[W] Fit failed for {}/{}: {} (function: {})'.format(state_or_tran, model_attribute, res.message, self._model_str))
        else:
            print('[W] Insufficient amount of valid parameter keys, cannot fit {}/{}'.format(state_or_tran, model_attribute))

    def is_predictable(self, param_list):
        for i, param in enumerate(param_list):
            if self._dependson[i] and not is_numeric(param):
                return False
        return True

    def eval(self, param_list):
        return self._function(self._regression_args, param_list)

class analytic:
    _num0_8 = np.vectorize(lambda x: 8 - bin(int(x)).count("1"))
    _num0_16 = np.vectorize(lambda x: 16 - bin(int(x)).count("1"))
    _num1 = np.vectorize(lambda x: bin(int(x)).count("1"))

    _function_map = {
        'linear' : lambda x: x,
        'logarithmic' : np.log,
        'logarithmic1' : lambda x: np.log(x + 1),
        'exponential' : np.exp,
        'square' : lambda x : x ** 2,
        'fractional' : lambda x : 1 / x,
        'sqrt' : np.sqrt,
        'num0_8' : _num0_8,
        'num0_16' : _num0_16,
        'num1' : _num1,
    }

    def functions():
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
            'fractional' : ParamFunction(
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
        if function_type == 'fractional':
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
        return AnalyticFunction(buf, arg_idx, parameter_names, num_args)

    #def function_powerset(function_descriptions):
    #    function_buffer = lambda param, arg: 0
    #    param_idx = 0
    #    for combination in powerset(function_descriptions):
    #        new_function = lambda param, arg: param[param_idx]
    #        param_idx += 1
    #        for function_name in combination:
    #            new_function = lambda param, arg: new_function(param, arg) * analytic._function_map[function_name](arg)
    #        new_function = lambda param, arg: param[param_idx] * 
    #        function_buffer = lambda param, arg: function_buffer(param, arg) + 

def _try_fits_parallel(arg):
    return {
        'key' : arg['key'],
        'result' : _try_fits(*arg['args'])
    }


def _try_fits(by_param, state_or_tran, model_attribute, param_index):
    functions = analytic.functions()


    for param_key in filter(lambda x: x[0] == state_or_tran, by_param.keys()):
        # We might remove elements from 'functions' while iterating over
        # its keys. A generator will not allow this, so we need to
        # convert to a list.
        function_names = list(functions.keys())
        for function_name in function_names:
            function_object = functions[function_name]
            if is_numeric(param_key[1][param_index]) and not function_object.is_valid(param_key[1][param_index]):
                functions.pop(function_name, None)

    raw_results = {}
    ref_results = {
        'mean' : [],
        'median' : []
    }
    results = {}

    for param_key in filter(lambda x: x[0] == state_or_tran, by_param.keys()):
        X = []
        Y = []
        num_valid = 0
        num_total = 0
        for k, v in by_param.items():
            if _param_slice_eq(k, param_key, param_index):
                num_total += 1
                if is_numeric(k[1][param_index]):
                    num_valid += 1
                    X.extend([float(k[1][param_index])] * len(v[model_attribute]))
                    Y.extend(v[model_attribute])

        if num_valid > 2:
            X = np.array(X)
            Y = np.array(Y)
            for function_name, param_function in functions.items():
                raw_results[function_name] = {}
                error_function = param_function.error_function
                res = optimize.least_squares(error_function, [0, 1], args=(X, Y), xtol=2e-15)
                measures = regression_measures(param_function.eval(res.x, X), Y)
                for measure, error_rate in measures.items():
                    if not measure in raw_results[function_name]:
                        raw_results[function_name][measure] = []
                    raw_results[function_name][measure].append(error_rate)
                #print(function_name, res, measures)
            mean_measures = aggregate_measures(np.mean(Y), Y)
            ref_results['mean'].append(mean_measures['rmsd'])
            median_measures = aggregate_measures(np.median(Y), Y)
            ref_results['median'].append(median_measures['rmsd'])

    best_fit_val = np.inf
    best_fit_name = None
    for function_name, result in raw_results.items():
        if len(result) > 0:
            results[function_name] = {}
            for measure in result.keys():
                results[function_name][measure] = np.mean(result[measure])
            rmsd = results[function_name]['rmsd']
            if rmsd < best_fit_val:
                best_fit_val = rmsd
                best_fit_name = function_name

    return {
        'best' : best_fit_name,
        'best_rmsd' : best_fit_val,
        'mean_rmsd' : np.mean(ref_results['mean']),
        'median_rmsd' : np.mean(ref_results['median']),
        'results' : results
    }

def _compute_param_statistics_parallel(args):
    return {
        'state_or_trans' : args['state_or_trans'],
        'key' : args['key'],
        'result' : _compute_param_statistics(*args['args'])
    }

def _compute_param_statistics(by_name, by_param, parameter_names, num_args, state_or_trans, key):
    ret = {
        'std_static' : np.std(by_name[state_or_trans][key]),
        'std_param_lut' : np.mean([np.std(by_param[x][key]) for x in by_param.keys() if x[0] == state_or_trans]),
        'std_by_param' : {},
        'std_by_arg' : [],
    }

    for param_idx, param in enumerate(parameter_names):
        ret['std_by_param'][param] = _mean_std_by_param(by_param, state_or_trans, key, param_idx)
    if arg_support_enabled and by_name[state_or_trans]['isa'] == 'transition':
        for arg_index in range(num_args[state_or_trans]):
            ret['std_by_arg'].append(_mean_std_by_param(by_param, state_or_trans, key, len(parameter_names) + arg_index))

    return ret

# returns the mean standard deviation of all measurements of 'what'
# (e.g. power consumption or timeout) for state/transition 'name' where
# parameter 'index' is dynamic and all other parameters are fixed.
# I.e., if parameters are a, b, c ∈ {1,2,3} and 'index' corresponds to b', then
# this function returns the mean of the standard deviations of (a=1, b=*, c=1),
# (a=1, b=*, c=2), and so on
def _mean_std_by_param(by_param, state_or_tran, key, param_index):
    partitions = []
    for param_value in filter(lambda x: x[0] == state_or_tran, by_param.keys()):
        param_partition = []
        for k, v in by_param.items():
            if _param_slice_eq(k, param_value, param_index):
                param_partition.extend(v[key])
        if len(param_partition):
            partitions.append(param_partition)
        else:
            print('[W] parameter value partition for {} is empty'.format(param_value))
    return np.mean([np.std(partition) for partition in partitions])

class EnergyModel:

    def __init__(self, preprocessed_data):
        self.traces = preprocessed_data
        self.by_name = {}
        self.by_param = {}
        self.by_trace = {}
        self.stats = {}
        np.seterr('raise')
        self._parameter_names = sorted(self.traces[0]['trace'][0]['parameter'].keys())
        self._num_args = {}
        for runidx, run in enumerate(self.traces):
            # if opts['ignore-trace-idx'] != runidx
            for i, elem in enumerate(run['trace']):
                if elem['name'] != 'UNINITIALIZED':
                    self._load_run_elem(i, elem)
                if elem['isa'] == 'transition' and not elem['name'] in self._num_args and 'args' in elem:
                    self._num_args[elem['name']] = len(elem['args'])
        self._aggregate_to_ndarray(self.by_name)
        self._compute_all_param_statistics()

    def _compute_all_param_statistics(self):
        queue = []
        for state_or_trans in self.by_name.keys():
            self.stats[state_or_trans] = {}
            for key in ['power', 'energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']:
                if key in self.by_name[state_or_trans]:
                    self.stats[state_or_trans][key] = _compute_param_statistics(self.by_name, self.by_param, self._parameter_names, self._num_args, state_or_trans, key)
                    #queue.append({
                    #    'state_or_trans' : state_or_trans,
                    #    'key' : key,
                    #    'args' : [self.by_name, self.by_param, self._parameter_names, self._num_args, state_or_trans, key]
                    #})

        # IPC overhead for by_name/by_param (un)pickling is higher than
        # multiprocessing speedup...  so let's not do this.
        #with Pool() as pool:
        #    results = pool.map(_compute_param_statistics_parallel, queue)
        #for ret in results:
        #    self.stats[ret['state_or_trans']][ret['key']] = ret['result']

    @classmethod
    def from_model(self, model_data, parameter_names):
        self.by_name = {}
        self.by_param = {}
        self.stats = {}
        np.seterr('raise')
        self._parameter_names = parameter_names
        for state_or_tran, values in model_data.items():
            for elem in values:
                self._load_agg_elem(state_or_tran, elem)
                #if elem['isa'] == 'transition' and not state_or_tran in self._num_args and 'args' in elem:
                #    self._num_args = len(elem['args'])
        self._aggregate_to_ndarray(self.by_name)
        self._compute_all_param_statistics()

    def _aggregate_to_ndarray(self, aggregate):
        for elem in aggregate.values():
            for key in ['power', 'power_std', 'energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']:
                if key in elem:
                    elem[key] = np.array(elem[key])


    def _add_data_to_aggregate(self, aggregate, key, element):
        if not key in aggregate:
            aggregate[key] = {
                'isa' : element['isa']
            }
            for datakey in element['offline_aggregates'].keys():
                aggregate[key][datakey] = []
        for datakey, dataval in element['offline_aggregates'].items():
            aggregate[key][datakey].extend(dataval)

    def _load_agg_elem(self, name, elem):
        self._add_data_to_aggregate(self.by_name, name, elem)
        self._add_data_to_aggregate(self.by_param, (name, tuple(elem['param'])), elem)

    def _load_run_elem(self, i, elem):
        self._add_data_to_aggregate(self.by_name, elem['name'], elem)
        self._add_data_to_aggregate(self.by_param, (elem['name'], tuple(_elem_param_and_arg_list(elem))), elem)

    def generic_param_independence_ratio(self, state_or_trans, key):
        statistics = self.stats[state_or_trans][key]
        if statistics['std_static'] == 0:
            return 0
        return statistics['std_param_lut'] / statistics['std_static']

    def generic_param_dependence_ratio(self, state_or_trans, key):
        return 1 - self.generic_param_independence_ratio(state_or_trans, key)

    def param_independence_ratio(self, state_or_trans, key, param):
        statistics = self.stats[state_or_trans][key]
        if statistics['std_by_param'][param] == 0:
            return 0
        return statistics['std_param_lut'] / statistics['std_by_param'][param]

    def param_dependence_ratio(self, state_or_trans, key, param):
        return 1 - self.param_independence_ratio(state_or_trans, key, param)

    def arg_independence_ratio(self, state_or_trans, key, arg_index):
        statistics = self.stats[state_or_trans][key]
        if statistics['std_by_arg'][arg_index] == 0:
            return 0
        return statistics['std_param_lut'] / statistics['std_by_arg'][arg_index]

    def arg_dependence_ratio(self, state_or_trans, key, arg_index):
        return 1 - self.arg_independence_ratio(state_or_trans, key, arg_index)

    def _get_model_from_dict(self, model_dict, model_function):
        model = {}
        for name, elem in model_dict.items():
            model[name] = {}
            for key in ['power', 'energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']:
                if key in elem:
                    try:
                        model[name][key] = model_function(elem[key])
                    except RuntimeWarning:
                        print('[W] Got no data for {} {}'.format(name, key))
                    except FloatingPointError as fpe:
                        print('[W] Got no data for {} {}: {}'.format(name, key, fpe))
        return model

    def get_static(self):
        static_model = self._get_model_from_dict(self.by_name, np.median)

        def static_median_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_median_getter

    def get_static_using_mean(self):
        static_model = self._get_model_from_dict(self.by_name, np.mean)

        def static_mean_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_mean_getter

    def get_param_lut(self):
        lut_model = self._get_model_from_dict(self.by_param, np.median)

        def lut_median_getter(name, key, param, arg = [], **kwargs):
            param.extend(map(_soft_cast_int, arg))
            return lut_model[(name, tuple(param))][key]

        return lut_median_getter

    def get_param_analytic(self):
        static_model = self._get_model_from_dict(self.by_name, np.median)

    def get_fitted(self):
        static_model = self._get_model_from_dict(self.by_name, np.median)
        param_model = dict([[state_or_tran, {}] for state_or_tran in self.by_name.keys()])
        fit_queue = []
        for state_or_tran in self.by_name.keys():
            param_keys = filter(lambda k: k[0] == state_or_tran, self.by_param.keys())
            param_subdict = dict(map(lambda k: [k, self.by_param[k]], param_keys))
            if self.by_name[state_or_tran]['isa'] == 'state':
                attributes = ['power']
            else:
                attributes = ['energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']
            for model_attribute in attributes:
                fit_results = {}
                for parameter_index, parameter_name in enumerate(self._parameter_names):
                    if self.param_dependence_ratio(state_or_tran, model_attribute, parameter_name) > 0.5:
                        fit_queue.append({
                            'key' : [state_or_tran, model_attribute, parameter_name],
                            'args' : [self.by_param, state_or_tran, model_attribute, parameter_index]
                        })
                        #fit_results[parameter_name] = _try_fits(self.by_param, state_or_tran, model_attribute, parameter_index)
                        #print('{} {} is {}'.format(state_or_tran, parameter_name, fit_results[parameter_name]['best']))
                if arg_support_enabled and self.by_name[state_or_tran]['isa'] == 'transition':
                    for arg_index in range(self._num_args[state_or_tran]):
                        if self.arg_dependence_ratio(state_or_tran, model_attribute, arg_index) > 0.5:
                            fit_queue.append({
                                'key' : [state_or_tran, model_attribute, arg_index],
                                'args' : [param_subdict, state_or_tran, model_attribute, len(self._parameter_names) + arg_index]
                            })
                            #fit_results[_arg_name(arg_index)] = _try_fits(self.by_param, state_or_tran, model_attribute, len(self._parameter_names) + arg_index)
                #if 'args' in self.by_name[state_or_tran]:
                #    for i, arg in range(len(self.by_name
        with Pool() as pool:
            all_fit_results = pool.map(_try_fits_parallel, fit_queue)

        for state_or_tran in self.by_name.keys():
            num_args = 0
            if arg_support_enabled and self.by_name[state_or_tran]['isa'] == 'transition':
                num_args = self._num_args[state_or_tran]
            if self.by_name[state_or_tran]['isa'] == 'state':
                attributes = ['power']
            else:
                attributes = ['energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']
            for model_attribute in attributes:
                fit_results = {}
                for result in all_fit_results:
                    if result['key'][0] == state_or_tran and result['key'][1] == model_attribute:
                        fit_result = result['result']
                        if fit_result['best_rmsd'] >= min(fit_result['mean_rmsd'], fit_result['median_rmsd']):
                            print('[I] Not modeling {} {} as function of {}: best ({:.0f}) is worse than ref ({:.0f}, {:.0f})'.format(
                                state_or_tran, model_attribute, result['key'][2], fit_result['best_rmsd'],
                                fit_result['mean_rmsd'], fit_result['median_rmsd']))
                        elif fit_result['best_rmsd'] >= 0.5 * min(fit_result['mean_rmsd'], fit_result['median_rmsd']):
                            print('[I] Not modeling {} {} as function o {}: best ({:.0f}) is not much better than ({:.0f}, {:.0f})'.format(
                                state_or_tran, model_attribute, result['key'][2], fit_result['best_rmsd'],
                                fit_result['mean_rmsd'], fit_result['median_rmsd']))
                        else:
                            fit_results[result['key'][2]] = fit_result

                if len(fit_results.keys()):
                    x = analytic.function_powerset(fit_results, self._parameter_names, num_args)
                    x.fit(self.by_param, state_or_tran, model_attribute)
                    param_model[state_or_tran][model_attribute] = {
                        'fit_result': fit_results,
                        'function' : x
                    }

        def model_getter(name, key, **kwargs):
            if key in param_model[name]:
                param_list = kwargs['param']
                param_function = param_model[name][key]['function']
                if param_function.is_predictable(param_list):
                    return param_function.eval(param_list)
            return static_model[name][key]

        def info_getter(name, key):
            if key in param_model[name]:
                return param_model[name][key]
            return None

        return model_getter, info_getter


    def states(self):
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'state', self.by_name.keys())))

    def transitions(self):
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'transition', self.by_name.keys())))

    def parameters(self):
        return self._parameter_names

    def assess(self, model_function):
        results = {}
        for name, elem in sorted(self.by_name.items()):
            results[name] = {}
            if elem['isa'] == 'state':
                predicted_data = np.array(list(map(lambda i: model_function(name, 'power', param=elem['param'][i]), range(len(elem['power'])))))
                measures = regression_measures(predicted_data, elem['power'])
                results[name]['power'] = measures
            else:
                for key in ['duration', 'energy', 'rel_energy_prev', 'rel_energy_next', 'timeout']:
                    predicted_data = np.array(list(map(lambda i: model_function(name, key, param=elem['param'][i]), range(len(elem[key])))))
                    measures = regression_measures(predicted_data, elem[key])
                    results[name][key] = measures
        return results



class MIMOSA:

    def __init__(self, voltage, shunt):
        self.voltage = voltage
        self.shunt = shunt
        self.r1 = 984 # "1k"
        self.r2 = 99013 # "100k"

    def charge_to_current_nocal(self, charge):
        ua_max = 1.836 / self.shunt * 1000000
        ua_step = ua_max / 65535
        return charge * ua_step

    def _load_tf(self, tf):
        num_bytes = tf.getmember('/tmp/mimosa//mimosa_scale_1.tmp').size
        charges = np.ndarray(shape=(int(num_bytes / 4)), dtype=np.int32)
        triggers = np.ndarray(shape=(int(num_bytes / 4)), dtype=np.int8)
        with tf.extractfile('/tmp/mimosa//mimosa_scale_1.tmp') as f:
            content = f.read()
            iterator = struct.iter_unpack('<I', content)
            i = 0
            for word in iterator:
                charges[i] = (word[0] >> 4)
                triggers[i] = (word[0] & 0x08) >> 3
                i += 1
        return charges, triggers


    def load_data(self, raw_data):
        with io.BytesIO(raw_data) as data_object:
            with tarfile.open(fileobj = data_object) as tf:
                return self._load_tf(tf)

    def currents_nocal(self, charges):
        ua_max = 1.836 / self.shunt * 1000000
        ua_step = ua_max / 65535
        return charges.astype(np.double) * ua_step

    def trigger_edges(self, triggers):
        trigidx = []
        prevtrig = triggers[0]
        # the device is reset for MIMOSA calibration in the first 10s and may
        # send bogus interrupts -> bogus triggers
        for i in range(1000000, triggers.shape[0]):
            trig = triggers[i]
            if trig != prevtrig:
                # Due to MIMOSA's integrate-read-reset cycle, the trigger
                # appears two points (20µs) before the corresponding data
                trigidx.append(i+2)
            prevtrig = trig
        return trigidx

    def calibration_edges(self, currents):
        r1idx = 0
        r2idx = 0
        ua_r1 = self.voltage / self.r1 * 1000000
        # first second may be bogus
        for i in range(100000, len(currents)):
            if r1idx == 0 and currents[i] > ua_r1 * 0.6:
                r1idx = i
            elif r1idx != 0 and r2idx == 0 and i > (r1idx + 180000) and currents[i] < ua_r1 * 0.4:
                r2idx = i
        # 2s disconnected, 2s r1, 2s r2  with r1 < r2  ->  ua_r1 > ua_r2
        # allow 5ms buffer in both directions to account for bouncing relais contacts
        return r1idx - 180500, r1idx - 500, r1idx + 500, r2idx - 500, r2idx + 500, r2idx + 180500

    def calibration_function(self, charges, cal_edges):
        dis_start, dis_end, r1_start, r1_end, r2_start, r2_end = cal_edges
        if dis_start < 0:
            dis_start = 0
        chg_r0 = charges[dis_start:dis_end]
        chg_r1 = charges[r1_start:r1_end]
        chg_r2 = charges[r2_start:r2_end]
        cal_0_mean = np.mean(chg_r0)
        cal_0_std = np.std(chg_r0)
        cal_r1_mean = np.mean(chg_r1)
        cal_r1_std = np.std(chg_r1)
        cal_r2_mean = np.mean(chg_r2)
        cal_r2_std = np.std(chg_r2)

        ua_r1 = self.voltage / self.r1 * 1000000
        ua_r2 = self.voltage / self.r2 * 1000000

        if cal_r2_mean > cal_0_mean:
            b_lower = (ua_r2 - 0) / (cal_r2_mean - cal_0_mean)
        else:
            print('[W] 0 uA == %.f uA during calibration' % (ua_r2))
            b_lower = 0

        b_upper = (ua_r1 - ua_r2) / (cal_r1_mean - cal_r2_mean)
        b_total = (ua_r1 - 0) / (cal_r1_mean - cal_0_mean)

        a_lower = -b_lower * cal_0_mean
        a_upper = -b_upper * cal_r2_mean
        a_total = -b_total * cal_0_mean

        if self.shunt == 680:
            # R1 current is higher than shunt range -> only use R2 for calibration
            def calfunc(charge):
                if charge < cal_0_mean:
                    return 0
                else:
                    return charge * b_lower + a_lower
        else:
            def calfunc(charge):
                if charge < cal_0_mean:
                    return 0
                if charge <= cal_r2_mean:
                    return charge * b_lower + a_lower
                else:
                    return charge * b_upper + a_upper + ua_r2

        caldata = {
            'edges' : [x * 10 for x in cal_edges],
            'offset': cal_0_mean,
            'offset2' : cal_r2_mean,
            'slope_low' : b_lower,
            'slope_high' : b_upper,
            'add_low' : a_lower,
            'add_high' : a_upper,
            'r0_err_uW' : np.mean(self.currents_nocal(chg_r0)) * self.voltage,
            'r0_std_uW' : np.std(self.currents_nocal(chg_r0)) * self.voltage,
            'r1_err_uW' : (np.mean(self.currents_nocal(chg_r1)) - ua_r1) * self.voltage,
            'r1_std_uW' : np.std(self.currents_nocal(chg_r1)) * self.voltage,
            'r2_err_uW' : (np.mean(self.currents_nocal(chg_r2)) - ua_r2) * self.voltage,
            'r2_std_uW' : np.std(self.currents_nocal(chg_r2)) * self.voltage,
        }

        #print("if charge < %f : return 0" % cal_0_mean)
        #print("if charge <= %f : return charge * %f + %f" % (cal_r2_mean, b_lower, a_lower))
        #print("else : return charge * %f + %f + %f" % (b_upper, a_upper, ua_r2))

        return calfunc, caldata

    def calcgrad(self, currents, threshold):
        grad = np.gradient(running_mean(currents * self.voltage, 10))
        # len(grad) == len(currents) - 9
        subst = []
        lastgrad = 0
        for i in range(len(grad)):
            # minimum substate duration: 10ms
            if np.abs(grad[i]) > threshold and i - lastgrad > 50:
                # account for skew introduced by running_mean and current
                # ramp slope (parasitic capacitors etc.)
                subst.append(i+10)
                lastgrad = i
        if lastgrad != i:
            subst.append(i+10)
        return subst

    # TODO konfigurierbare min/max threshold und len(gradidx) > X, binaere
    # Sache nach noetiger threshold. postprocessing mit
    # "zwei benachbarte substates haben sehr aehnliche werte / niedrige stddev" -> mergen
    # ... min/max muessen nicht vorgegeben werden, sind ja bekannt (0 / np.max(grad))
    # TODO bei substates / index foo den offset durch running_mean beachten
    # TODO ggf. clustering der 'abs(grad) > threshold' und bestimmung interessanter
    # uebergaenge dadurch?
    def gradfoo(self, currents):
        gradients = np.abs(np.gradient(running_mean(currents * self.voltage, 10)))
        gradmin = np.min(gradients)
        gradmax = np.max(gradients)
        threshold = np.mean([gradmin, gradmax])
        gradidx = self.calcgrad(currents, threshold)
        num_substates = 2
        while len(gradidx) != num_substates:
            if gradmax - gradmin < 0.1:
                # We did our best
                return threshold, gradidx
            if len(gradidx) > num_substates:
                gradmin = threshold
            else:
                gradmax = threshold
            threshold = np.mean([gradmin, gradmax])
            gradidx = self.calcgrad(currents, threshold)
        return threshold, gradidx

    def analyze_states(self, charges, trigidx, ua_func):
        previdx = 0
        is_state = True
        iterdata = []
        for idx in trigidx:
            range_raw = charges[previdx:idx]
            range_ua = ua_func(range_raw)
            substates = {}

            if previdx != 0 and idx - previdx > 200:
                thr, subst = 0, [] #self.gradfoo(range_ua)
                if len(subst):
                    statelist = []
                    prevsubidx = 0
                    for subidx in subst:
                        statelist.append({
                            'duration': (subidx - prevsubidx) * 10,
                            'uW_mean' : np.mean(range_ua[prevsubidx : subidx] * self.voltage),
                            'uW_std'  : np.std(range_ua[prevsubidx : subidx] * self.voltage),
                        })
                        prevsubidx = subidx
                    substates = {
                        'threshold' : thr,
                        'states' : statelist,
                    }

            isa = 'state'
            if not is_state:
                isa = 'transition'

            data = {
                'isa': isa,
                'clip_rate' : np.mean(range_raw == 65535),
                'raw_mean': np.mean(range_raw),
                'raw_std' : np.std(range_raw),
                'uW_mean' : np.mean(range_ua * self.voltage),
                'uW_std' : np.std(range_ua * self.voltage),
                'us' : (idx - previdx) * 10,
            }

            if 'states' in substates:
                data['substates'] = substates
                ssum = np.sum(list(map(lambda x : x['duration'], substates['states'])))
                if ssum != data['us']:
                    print("ERR: duration %d vs %d" % (data['us'], ssum))

            if isa == 'transition':
                # subtract average power of previous state
                # (that is, the state from which this transition originates)
                data['uW_mean_delta_prev'] = data['uW_mean'] - iterdata[-1]['uW_mean']
                # placeholder to avoid extra cases in the analysis
                data['uW_mean_delta_next'] = data['uW_mean']
                data['timeout'] = iterdata[-1]['us']
            elif len(iterdata) > 0:
                # subtract average power of next state
                # (the state into which this transition leads)
                iterdata[-1]['uW_mean_delta_next'] = iterdata[-1]['uW_mean'] - data['uW_mean']

            iterdata.append(data)

            previdx = idx
            is_state = not is_state
        return iterdata
