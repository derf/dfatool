#!/usr/bin/env python3

import csv
import io
import json
import numpy as np
import os
import re
from scipy import optimize
from sklearn.metrics import r2_score
import struct
import sys
import tarfile
from multiprocessing import Pool
from automata import PTA
from functions import analytic
from functions import AnalyticFunction
from utils import *

arg_support_enabled = True

def running_mean(x, N):
    """
    Compute running average.

    arguments:
    x -- NumPy array
    N -- how many items to average
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

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

def vprint(verbose, string):
    """
    Print string if verbose.

    Prints string if verbose is a True value
    """
    if verbose:
        print(string)

    # I don't recall what these are for.
    # --df, 2019-01-29
    def _gplearn_add_(x, y):
        return x + y

    def _gplearn_sub_(x, y):
        return x - y

    def _gplearn_mul_(x, y):
        return x * y

    def _gplearn_div_(x, y):
        if np.abs(y) > 0.001:
            return x / y
        return 1.

def gplearn_to_function(function_str):
    """
    Convert gplearn-style function string to Python function.

    Takes a function string like "mul(add(X0, X1), X2)" and returns
    a Python function implementing the specified behaviour,
    e.g. "lambda x, y, z: (x + y) * z".

    Supported functions:
    add  --  x + y
    sub  --  x - y
    mul  --  x * y
    div  --  x / y if |y| > 0.001, otherwise 1
    sqrt --  sqrt(|x|)
    log  --  log(|x|) if |x| > 0.001, otherwise 0
    inv  --  1 / x if |x| > 0.001, otherwise 0
    """
    eval_globals = {
        'add' : lambda x, y : x + y,
        'sub' : lambda x, y : x - y,
        'mul' : lambda x, y : x * y,
        'div' : lambda x, y : np.divide(x, y) if np.abs(y) > 0.001 else 1.,
        'sqrt': lambda x : np.sqrt(np.abs(x)),
        'log' : lambda x : np.log(np.abs(x)) if np.abs(x) > 0.001 else 0.,
        'inv' : lambda x : 1. / x if np.abs(x) > 0.001 else 0.,
    }

    last_arg_index = 0
    for i in range(0, 100):
        if function_str.find('X{:d}'.format(i)) >= 0:
            last_arg_index = i

    arg_list = []
    for i in range(0, last_arg_index+1):
        arg_list.append('X{:d}'.format(i))

    eval_str = 'lambda {}, *whatever: {}'.format(','.join(arg_list), function_str)
    print(eval_str)
    return eval(eval_str, eval_globals)

def _elem_param_and_arg_list(elem):
    param_dict = elem['parameter']
    paramkeys = sorted(param_dict.keys())
    paramvalue = [soft_cast_int(param_dict[x]) for x in paramkeys]
    if arg_support_enabled and 'args' in elem:
        paramvalue.extend(map(soft_cast_int, elem['args']))
    return paramvalue

def _arg_name(arg_index):
    return '~arg{:02}'.format(arg_index)

def append_if_set(aggregate, data, key):
    """Append data[key] to aggregate if key in data."""
    if key in data:
        aggregate.append(data[key])

def mean_or_none(arr):
    """Compute mean of NumPy array arr, return -1 if empty."""
    if len(arr):
        return np.mean(arr)
    return -1

def aggregate_measures(aggregate, actual):
    """
    Calculate error measures for model value on data list.

    arguments:
    aggregate -- model value (float or int)
    actual -- real-world / reference values (list of float or int)

    return value:
    See regression_measures
    """
    aggregate_array = np.array([aggregate] * len(actual))
    return regression_measures(aggregate_array, np.array(actual))

def regression_measures(predicted, actual):
    """
    Calculate error measures by comparing model values to reference values.

    arguments:
    predicted -- model values (np.ndarray)
    actual -- real-world / reference values (np.ndarray)

    Returns a dict containing the following measures:
    mae -- Mean Absolute Error
    mape -- Mean Absolute Percentage Error,
            if all items in actual are non-zero (NaN otherwise)
    smape -- Symmetric Mean Absolute Percentage Error,
             if no 0,0-pairs are present in actual and predicted (NaN otherwise)
    msd -- Mean Square Deviation
    rmsd -- Root Mean Square Deviation
    ssr -- Sum of Squared Residuals
    rsq -- R^2 measure, see sklearn.metrics.r2_score
    count -- Number of values
    """
    if type(predicted) != np.ndarray:
        raise ValueError('first arg must be ndarray, is {}'.format(type(predicted)))
    if type(actual) != np.ndarray:
        raise ValueError('second arg must be ndarray, is {}'.format(type(actual)))
    deviations = predicted - actual
    mean = np.mean(actual)
    if len(deviations) == 0:
        return {}
    measures = {
        'mae' : np.mean(np.abs(deviations), dtype=np.float64),
        'msd' : np.mean(deviations**2, dtype=np.float64),
        'rmsd' : np.sqrt(np.mean(deviations**2), dtype=np.float64),
        'ssr' : np.sum(deviations**2, dtype=np.float64),
        'rsq' : r2_score(actual, predicted),
        'count' : len(actual),
    }

    #rsq_quotient = np.sum((actual - mean)**2, dtype=np.float64) * np.sum((predicted - mean)**2, dtype=np.float64)

    if np.all(actual != 0):
        measures['mape'] = np.mean(np.abs(deviations / actual)) * 100 # bad measure
    else:
        measures['mape'] = np.nan
    if np.all(np.abs(predicted) + np.abs(actual) != 0):
        measures['smape'] = np.mean(np.abs(deviations) / (( np.abs(predicted) + np.abs(actual)) / 2 )) * 100
    else:
        measures['smape'] = np.nan
    #if np.all(rsq_quotient != 0):
    #    measures['rsq'] = (np.sum((actual - mean) * (predicted - mean), dtype=np.float64)**2) / rsq_quotient

    return measures

class KeysightCSV:
    """Simple loader for Keysight CSV data, as exported by the windows software."""

    def __init__(self):
        """Create a new KeysightCSV object."""
        pass

    def load_data(self, filename):
        """
        Load log data from filename, return timestamps and currents.

        Returns two one-dimensional NumPy arrays: timestamps and corresponding currents.
        """
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

def _xv_partitions_kfold(length, num_slices):
    pairs = []
    indexes = np.arange(length)
    for i in range(0, num_slices):
        training = np.delete(indexes, slice(i, None, num_slices))
        validation = indexes[i::num_slices]
        pairs.append((training, validation))
    return pairs

def _xv_partitions_montecarlo(length, num_slices):
    pairs = []
    for i in range(0, num_slices):
        shuffled = np.random.permutation(np.arange(length))
        border = int(length * float(2) / 3)
        training = shuffled[:border]
        validation = shuffled[border:]
        pairs.append((training, validation))
    return pairs

class CrossValidation:

    def __init__(self, em, num_partitions):
        self._em = em
        self._num_partitions = num_partitions
        x = EnergyModel.from_model(em.by_name, em._parameter_names)


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

class ParamStats:

    def __init__(self, by_name, by_param, parameter_names, arg_count, use_corrcoef = False):
        """
        Compute standard deviation and correlation coefficient on parameterized data partitions.

        It is strongly recommended to vary all parameter values evenly.
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
        use_corrcoef -- use correlation coefficient instead of stddev heuristic for parameter detection
        """
        self.stats = dict()
        self.use_corrcoef = use_corrcoef
        # Note: This is deliberately single-threaded. The overhead incurred
        # by multiprocessing is higher than the speed gained by parallel
        # computation of statistics measures.
        for state_or_tran in by_name.keys():
            self.stats[state_or_tran] = dict()
            for attribute in by_name[state_or_tran]['attributes']:
                self.stats[state_or_tran][attribute] = compute_param_statistics(by_name, by_param, parameter_names, arg_count, state_or_tran, attribute)

    def _generic_param_independence_ratio(self, state_or_trans, attribute):
        """
        Return the heuristic ratio of parameter independence for state_or_trans and attribute.

        This is not supported if the correlation coefficient is used.
        A value close to 1 means no influence, a value close to 0 means high probability of influence.
        """
        statistics = self.stats[state_or_trans][attribute]
        if self.use_corrcoef:
            # not supported
            raise ValueError
        if statistics['std_static'] == 0:
            return 0
        return statistics['std_param_lut'] / statistics['std_static']

    def generic_param_dependence_ratio(self, state_or_trans, attribute):
        """
        Return the heuristic ratio of parameter dependence for state_or_trans and attribute.

        This is not supported if the correlation coefficient is used.
        A value close to 0 means no influence, a value close to 1 means high probability of influence.
        """
        return 1 - self._generic_param_independence_ratio(state_or_trans, attribute)

    def _param_independence_ratio(self, state_or_trans, attribute, param):
        """
        Return the heuristic ratio of parameter independence for state_or_trans, attribute, and param.

        A value close to 1 means no influence, a value close to 0 means high probability of influence.
        """
        statistics = self.stats[state_or_trans][attribute]
        if self.use_corrcoef:
            return 1 - np.abs(statistics['corr_by_param'][param])
        if statistics['std_by_param'][param] == 0:
            if statistics['std_param_lut'] != 0:
                raise RuntimeError("wat")
            # In general, std_param_lut < std_by_param. So, if std_by_param == 0, std_param_lut == 0 follows.
            # This means that the variation of param does not affect the model quality -> no influence, return 1
            return 1
        return statistics['std_param_lut'] / statistics['std_by_param'][param]

    def param_dependence_ratio(self, state_or_trans, attribute, param):
        """
        Return the heuristic ratio of parameter dependence for state_or_trans, attribute, and param.

        A value close to 0 means no influence, a value close to 1 means high probability of influence.
        """
        return 1 - self._param_independence_ratio(state_or_trans, attribute, param)

    def _arg_independence_ratio(self, state_or_trans, attribute, arg_index):
        statistics = self.stats[state_or_trans][attribute]
        if self.use_corrcoef:
            return 1 - np.abs(statistics['corr_by_arg'][arg_index])
        if statistics['std_by_arg'][arg_index] == 0:
            if statistics['std_arg_lut'] != 0:
                raise RuntimeError("wat")
            # In general, std_arg_lut < std_by_arg. So, if std_by_arg == 0, std_arg_lut == 0 follows.
            # This means that the variation of arg does not affect the model quality -> no influence, return 1
            return 1
        return statistics['std_param_lut'] / statistics['std_by_arg'][arg_index]

    def arg_dependence_ratio(self, state_or_trans, attribute, arg_index):
        return 1 - self._arg_independence_ratio(state_or_trans, attribute, arg_index)

    # This heuristic is very similar to the "function is not much better than
    # median" checks in get_fitted. So far, doing it here as well is mostly
    # a performance and not an algorithm quality decision.
    # --df, 2018-04-18
    def depends_on_param(self, state_or_trans, attribute, param):
        """Return whether attribute of state_or_trans depens on param."""
        if self.use_corrcoef:
            return self.param_dependence_ratio(state_or_trans, attribute, param) > 0.1
        else:
            return self.param_dependence_ratio(state_or_trans, attribute, param) > 0.5

    # See notes on depends_on_param
    def depends_on_arg(self, state_or_trans, attribute, arg_index):
        """Return whether attribute of state_or_trans depens on arg_index."""
        if self.use_corrcoef:
            return self.arg_dependence_ratio(state_or_trans, attribute, arg_index) > 0.1
        else:
            return self.arg_dependence_ratio(state_or_trans, attribute, arg_index) > 0.5

class RawData:
    """
    Loader for hardware model traces measured with MIMOSA.

    Expects a specific trace format and UART log output (as produced by the
    dfatool benchmark generator). Loads data, prunes bogus measurements, and
    provides preprocessed data suitable for EnergyModel.
    """

    def __init__(self, filenames):
        """
        Create a new RawData object.

        Each filename element corresponds to a measurement run.
        """
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
            paramvalue = [soft_cast_int(online_trace_part['parameter'][x]) for x in paramkeys]

            # NB: Unscheduled transitions do not have an 'args' field set.
            # However, they should only be caused by interrupts, and
            # interrupts don't have args anyways.
            if arg_support_enabled and 'args' in online_trace_part:
                paramvalue.extend(map(soft_cast_int, online_trace_part['args']))

            if not 'offline_aggregates' in online_trace_part:
                online_trace_part['offline_attributes'] = ['power', 'duration', 'energy']
                online_trace_part['offline_aggregates'] = {
                    'power' : [],
                    'duration' : [],
                    'power_std' : [],
                    'energy' : [],
                    'paramkeys' : [],
                    'param': [],
                }
                if online_trace_part['isa'] == 'transition':
                    online_trace_part['offline_attributes'].extend(['rel_energy_prev', 'rel_energy_next', 'timeout'])
                    online_trace_part['offline_aggregates']['rel_energy_prev'] = []
                    online_trace_part['offline_aggregates']['rel_energy_next'] = []
                    online_trace_part['offline_aggregates']['timeout'] = []

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
                online_trace_part['offline_aggregates']['rel_energy_prev'].append(
                    offline_trace_part['uW_mean_delta_prev'] * (offline_trace_part['us'] - 20))
                online_trace_part['offline_aggregates']['rel_energy_next'].append(
                    offline_trace_part['uW_mean_delta_next'] * (offline_trace_part['us'] - 20))
                online_trace_part['offline_aggregates']['timeout'].append(
                    offline_trace_part['timeout'])

    def _concatenate_analyzed_traces(self):
        self.traces = []
        for trace in self.traces_by_fileno:
            self.traces.extend(trace)
        for i, trace in enumerate(self.traces):
            trace['orig_id'] = trace['id']
            trace['id'] = i

    def get_preprocessed_data(self, verbose = True):
        """
        Return a list of DFA traces annotated with energy, timing, and parameter data.

        Suitable for the EnergyModel constructor.
        See EnergyModel(...) docstring for format details.
        """
        self.verbose = verbose
        if self.preprocessed:
            return self.traces
        if self.version == 0:
            self._preprocess_0()
        self.preprocessed = True
        return self.traces

    def _preprocess_0(self):
        """Load raw MIMOSA data and turn it into measurements which are ready to be analyzed."""
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
            else:
                vprint(self.verbose, '[W] Skipping {ar:s}/{m:s}: {e:s}'.format(
                    ar = self.filenames[measurement['fileno']],
                    m = measurement['info'].name,
                    e = measurement['error']))
        vprint(self.verbose, '[I] {num_valid:d}/{num_total:d} measurements are valid'.format(
            num_valid = num_valid,
            num_total = len(measurements)))
        self._concatenate_analyzed_traces()
        self.preprocessing_stats = {
            'num_runs' : len(measurements),
            'num_valid' : num_valid
        }


def _try_fits_parallel(arg):
    return {
        'key' : arg['key'],
        'result' : _try_fits(*arg['args'])
    }


def _try_fits(by_param, state_or_tran, model_attribute, param_index, safe_functions_enabled = False):
    functions = analytic.functions(safe_functions_enabled = safe_functions_enabled)


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
            if param_slice_eq(k, param_key, param_index):
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

    if not len(ref_results['mean']):
        # Insufficient data for fitting
        return {
            'best' : None,
            'best_rmsd' : np.inf,
            'results' : results,
        }

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

class EnergyModel:
    u"""
    parameter-aware PTA-based energy model.

    Supports both static and parameter-based model attributes, and automatic detection of parameter-dependence.

    The model heavily relies on two internal data structures:
    EnergyModel.by_name and EnergyModel.by_param.

    These provide measurements aggregated by state/transition name
    and (in case of by_para) parameter values. Layout:
    dictionary with one key per state/transition ('send', 'TX', ...) or
    one key per state/transition and parameter combination
    (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).
    For by_param, parameter values are ordered corresponding to the lexically sorted parameter names.

    Each element is in turn a dict with the following elements:
    - isa: 'state' or 'transition'
    - power: list of mean power measurements in µW
    - duration: list of durations in µs
    - power_std: list of stddev of power per state/transition
    - energy: consumed energy (power*duration) in pJ
    - paramkeys: list of parameter names in each measurement (-> list of lists)
    - param: list of parameter values in each measurement (-> list of lists)
    - attributes: list of keys that should be analyzed,
        e.g. ['power', 'duration']
    additionally, only if isa == 'transition':
    - timeout: list of duration of previous state in µs
    - rel_energy_prev: transition energy relative to previous state mean power in pJ
    - rel_energy_next: transition energy relative to next state mean power in pJ
    """

    def __init__(self, preprocessed_data, ignore_trace_indexes = [], discard_outliers = None, function_override = {}, verbose = True, use_corrcoef = False, hwmodel = None):
        """
        Prepare a new PTA energy model.

        Actual model generation is done on-demand by calling the respective functions.

        arguments:
        preprocessed_data -- list of preprocessed DFA traces.
        ignore_trace_indexes -- list of trace indexes. The corresponding taces will be ignored.
        discard_outliers -- experimental: threshold for outlier detection and removel (float).
            Outlier detection is performed individually for each state/transition in each trace,
            so it only works if the benchmark ran several times.
            Given "data" (a set of measurements of the same thing, e.g. TX duration in the third benchmark trace),
            "m" (the median of all attribute measurements with the same parameters, which may include data from other traces),
            a data point X is considered an outlier if
            | 0.6745 * (X - m) / median(|data - m|) | > discard_outliers .
        function_override -- dict of overrides for automatic parameter function generation.
            If (state or transition name, model attribute) is present in function_override,
            the corresponding text string is the function used for analytic (parameter-aware/fitted)
            modeling of this attribute. It is passed to AnalyticFunction, see
            there for the required format. Note that this happens regardless of
            parameter dependency detection: The provided analytic function will be assigned
            even if it seems like the model attribute is static / parameter-independent.
        verbose -- print informative output, e.g. when removing an outlier
        use_corrcoef -- use correlation coefficient instead of stddev comparison
            to detect whether a model attribute depends on a parameter
        hwmodel -- hardware model suitable for PTA.from_hwmodel

        Detailed layout of preprocessed_data:
        [ ... Liste von einzelnen Läufen (d.h. eine Zustands- und Transitionsfolge UNINITIALIZED -> foo -> FOO -> bar -> BAR -> ...)
            Jeder Lauf:
            - id: int Nummer des Laufs, beginnend bei 1
            - trace: [ ... Liste von Zuständen und Transitionen
                Jeweils:
                - name: str Name
                - isa: str state // transition
                - parameter: { ... globaler Parameter: aktueller wert. null falls noch nicht eingestellt }
                - plan:
                    Falls isa == 'state':
                    - power: int(uW?)
                    - time: int(us) geplante Dauer
                    - energy: int(pJ?)
                    Falls isa == 'transition':
                    - timeout: int(us) oder null
                    - energy: int (pJ?)
                    - level: str 'user' 'epilogue'
                - offline_attributes: [ ... Namen der in offline_aggregates gespeicherten Modellattribute, z.B. param, duration, energy, timeout ]
                - offline_aggregates:
                    - power: [float(uW)] Mittlere Leistung während Zustand/Transitions
                    - power_std: [float(uW^2)] Standardabweichung der Leistung
                    - duration: [int(us)] Dauer
                    - energy: [float(pJ)] Energieaufnahme des Zustands / der Transition
                    - clip_rate: [float(0..1)] Clipping
                    - paramkeys: [[str]] Name der berücksichtigten Parameter
                    - param: [int // str] Parameterwerte. Quasi-Duplikat von 'parameter' oben
                    Falls isa == 'transition':
                    - timeout: [int(us)] Dauer des vorherigen Zustands
                    - rel_energy_prev: [int(pJ)]
                    - rel_energy_next: [int(pJ)]
                - offline: [ ... Während der Messung von MIMOSA o.ä. gemessene Werte
                    -> siehe doc/MIMOSA analyze_states
                    - isa: 'state' oder 'transition'
                    - clip_rate: range(0..1) Anteil an Clipping im Energieverbrauch
                    - raw_mean: Mittelwert der Rohwerte
                    - raw_std: Standardabweichung der Rohwerte
                    - uW_mean: Mittelwert der (kalibrierten) Leistungsaufnahme
                    - uW_std: Standardabweichung der (kalibrierten) Leistungsaufnahme
                    - us: Dauer
                    Nur falls isa 'transition':
                    - timeout: Dauer des vorherigen Zustands
                    - uW_mean_delta_prev
                    - uW_mean_delta_next
                ]
                - online: [ ... Während der Messung vom Betriebssystem bestimmte Daten
                    Falls isa == 'state':
                    - power: int(uW?)
                    - time: int(us) geplante Dauer
                    - energy: int(pJ?)
                    Falls isa == 'transition':
                    - timeout: int(us) oder null
                    - energy: int (pJ?)
                    - level: str ('user' oder 'epilogue')
                ]
                Falls isa == 'transition':
                - code: [str] Name und Argumente der aufgerufenen Funktion
                - args: [str] Argumente der aufgerufenen Funktion
            ]
        ]
        """
        self.traces = preprocessed_data
        self.by_name = {}
        self.by_param = {}
        self.by_trace = {}
        self.cache = {}
        np.seterr('raise')
        self._parameter_names = sorted(self.traces[0]['trace'][0]['parameter'].keys())
        self._num_args = {}
        self._outlier_threshold = discard_outliers
        self._use_corrcoef = use_corrcoef
        self.function_override = function_override
        self.verbose = verbose
        self.hwmodel = hwmodel
        self.ignore_trace_indexes = ignore_trace_indexes
        if discard_outliers != None:
            self._compute_outlier_stats(ignore_trace_indexes, discard_outliers)
        for run in self.traces:
            if run['id'] not in ignore_trace_indexes:
                for i, elem in enumerate(run['trace']):
                    if elem['name'] != 'UNINITIALIZED':
                        self._load_run_elem(i, elem)
                    if elem['isa'] == 'transition' and not elem['name'] in self._num_args and 'args' in elem:
                        self._num_args[elem['name']] = len(elem['args'])
        self._aggregate_to_ndarray(self.by_name)
        self._compute_all_param_statistics()

    def distinct_param_values(self, state_or_tran, param_index = None, arg_index = None):
        if param_index != None:
            param_values = map(lambda x: x[param_index], self.by_name[state_or_tran]['param'])
        return sorted(set(param_values))

    def _compute_outlier_stats(self, ignore_trace_indexes, threshold):
        tmp_by_param = {}
        self.median_by_param = {}
        for run in self.traces:
            if run['id'] not in ignore_trace_indexes:
                for i, elem in enumerate(run['trace']):
                    key = (elem['name'], tuple(_elem_param_and_arg_list(elem)))
                    if not key in tmp_by_param:
                        tmp_by_param[key] = {}
                        for attribute in elem['offline_attributes']:
                            tmp_by_param[key][attribute] = []
                    for attribute in elem['offline_attributes']:
                        tmp_by_param[key][attribute].extend(elem['offline_aggregates'][attribute])
        for key, elem in tmp_by_param.items():
            if not key in self.median_by_param:
                self.median_by_param[key] = {}
            for attribute in tmp_by_param[key].keys():
                self.median_by_param[key][attribute] = np.median(tmp_by_param[key][attribute])


    def _compute_all_param_statistics(self):
        self.stats = ParamStats(self.by_name, self.by_param, self._parameter_names, self._num_args, self._use_corrcoef)

    @classmethod
    def from_model(self, model_data, parameter_names):
        self.by_name = {}
        self.by_param = {}
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
            for key in elem['attributes']:
                elem[key] = np.array(elem[key])

    def _prune_outliers(self, key, attribute, data):
        if self._outlier_threshold == None:
            return data
        median = self.median_by_param[key][attribute]
        if np.median(np.abs(data - median)) == 0:
            return data
        pruned_data = list(filter(lambda x: np.abs(0.6745 * (x - median) / np.median(np.abs(data - median))) > self._outlier_threshold, data ))
        if len(pruned_data):
            vprint(self.verbose, '[I] Pruned outliers from ({}) {}: {}'.format(key, attribute, pruned_data))
            data = list(filter(lambda x: np.abs(0.6745 * (x - median) / np.median(np.abs(data - median))) <= self._outlier_threshold, data ))
        return data

    def _add_data_to_aggregate(self, aggregate, key, element):
        if not key in aggregate:
            aggregate[key] = {
                'isa' : element['isa']
            }
            for datakey in element['offline_aggregates'].keys():
                aggregate[key][datakey] = []
            if element['isa'] == 'state':
                aggregate[key]['attributes'] = ['power']
            else:
                aggregate[key]['attributes'] = ['duration', 'energy', 'rel_energy_prev', 'rel_energy_next']
                if element['plan']['level'] == 'epilogue':
                    aggregate[key]['attributes'].insert(0, 'timeout')
        for datakey, dataval in element['offline_aggregates'].items():
            if datakey in element['offline_attributes']:
                dataval = self._prune_outliers((element['name'], tuple(_elem_param_and_arg_list(element))), datakey, dataval)
            aggregate[key][datakey].extend(dataval)

    def _load_agg_elem(self, name, elem):
        self._add_data_to_aggregate(self.by_name, name, elem)
        self._add_data_to_aggregate(self.by_param, (name, tuple(elem['param'])), elem)

    def _load_run_elem(self, i, elem):
        self._add_data_to_aggregate(self.by_name, elem['name'], elem)
        self._add_data_to_aggregate(self.by_param, (elem['name'], tuple(_elem_param_and_arg_list(elem))), elem)

    # This heuristic is very similar to the "function is not much better than
    # median" checks in get_fitted. So far, doing it here as well is mostly
    # a performance and not an algorithm quality decision.
    # --df, 2018-04-18
    def depends_on_param(self, state_or_trans, key, param):
        return self.stats.depends_on_param(state_or_trans, key, param)

    # See notes on depends_on_param
    def depends_on_arg(self, state_or_trans, key, param):
        return self.stats.depends_on_arg(state_or_trans, key, param)

    def _get_model_from_dict(self, model_dict, model_function):
        model = {}
        for name, elem in model_dict.items():
            model[name] = {}
            for key in elem['attributes']:
                try:
                    model[name][key] = model_function(elem[key])
                except RuntimeWarning:
                    vprint(self.verbose, '[W] Got no data for {} {}'.format(name, key))
                except FloatingPointError as fpe:
                    vprint(self.verbose, '[W] Got no data for {} {}: {}'.format(name, key, fpe))
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
            param.extend(map(soft_cast_int, arg))
            return lut_model[(name, tuple(param))][key]

        return lut_median_getter

    def get_param_analytic(self):
        static_model = self._get_model_from_dict(self.by_name, np.median)

    def param_index(self, param_name):
        if param_name in self._parameter_names:
            return self._parameter_names.index(param_name)
        return len(self._parameter_names) + int(param_name)

    def param_name(self, param_index):
        if param_index < len(self._parameter_names):
            return self._parameter_names[param_index]
        return str(param_index)

    def get_fitted(self, safe_functions_enabled = False):

        if 'fitted_model_getter' in self.cache and 'fitted_info_getter' in self.cache:
            return self.cache['fitted_model_getter'], self.cache['fitted_info_getter']

        static_model = self._get_model_from_dict(self.by_name, np.median)
        param_model = dict([[state_or_tran, {}] for state_or_tran in self.by_name.keys()])
        fit_queue = []
        for state_or_tran in self.by_name.keys():
            param_keys = filter(lambda k: k[0] == state_or_tran, self.by_param.keys())
            param_subdict = dict(map(lambda k: [k, self.by_param[k]], param_keys))
            for model_attribute in self.by_name[state_or_tran]['attributes']:
                fit_results = {}
                for parameter_index, parameter_name in enumerate(self._parameter_names):
                    if self.depends_on_param(state_or_tran, model_attribute, parameter_name):
                        fit_queue.append({
                            'key' : [state_or_tran, model_attribute, parameter_name],
                            'args' : [self.by_param, state_or_tran, model_attribute, parameter_index, safe_functions_enabled]
                        })
                        #fit_results[parameter_name] = _try_fits(self.by_param, state_or_tran, model_attribute, parameter_index)
                        #print('{} {} is {}'.format(state_or_tran, parameter_name, fit_results[parameter_name]['best']))
                if arg_support_enabled and self.by_name[state_or_tran]['isa'] == 'transition':
                    for arg_index in range(self._num_args[state_or_tran]):
                        if self.depends_on_arg(state_or_tran, model_attribute, arg_index):
                            fit_queue.append({
                                'key' : [state_or_tran, model_attribute, arg_index],
                                'args' : [param_subdict, state_or_tran, model_attribute, len(self._parameter_names) + arg_index, safe_functions_enabled]
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
            for model_attribute in self.by_name[state_or_tran]['attributes']:
                fit_results = {}
                for result in all_fit_results:
                    if result['key'][0] == state_or_tran and result['key'][1] == model_attribute:
                        fit_result = result['result']
                        if fit_result['best_rmsd'] >= min(fit_result['mean_rmsd'], fit_result['median_rmsd']):
                            vprint(self.verbose, '[I] Not modeling {} {} as function of {}: best ({:.0f}) is worse than ref ({:.0f}, {:.0f})'.format(
                                state_or_tran, model_attribute, result['key'][2], fit_result['best_rmsd'],
                                fit_result['mean_rmsd'], fit_result['median_rmsd']))
                        # See notes on depends_on_param
                        elif fit_result['best_rmsd'] >= 0.8 * min(fit_result['mean_rmsd'], fit_result['median_rmsd']):
                            vprint(self.verbose, '[I] Not modeling {} {} as function of {}: best ({:.0f}) is not much better than ({:.0f}, {:.0f})'.format(
                                state_or_tran, model_attribute, result['key'][2], fit_result['best_rmsd'],
                                fit_result['mean_rmsd'], fit_result['median_rmsd']))
                        else:
                            fit_results[result['key'][2]] = fit_result

                if (state_or_tran, model_attribute) in self.function_override:
                    function_str = self.function_override[(state_or_tran, model_attribute)]
                    x = AnalyticFunction(function_str, self._parameter_names, num_args)
                    x.fit(self.by_param, state_or_tran, model_attribute)
                    if x.fit_success:
                        param_model[state_or_tran][model_attribute] = {
                            'fit_result': fit_results,
                            'function' : x
                        }
                elif len(fit_results.keys()):
                    x = analytic.function_powerset(fit_results, self._parameter_names, num_args)
                    x.fit(self.by_param, state_or_tran, model_attribute)
                    if x.fit_success:
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

        self.cache['fitted_model_getter'] = model_getter
        self.cache['fitted_info_getter'] = info_getter

        return model_getter, info_getter

    def to_json(self):
        static_model = self.get_static()
        _, param_info = self.get_fitted()
        pta = PTA.from_json(self.hwmodel)
        pta.update(static_model, param_info)
        return pta.to_json()

    def states(self):
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'state', self.by_name.keys())))

    def transitions(self):
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'transition', self.by_name.keys())))

    def states_and_transitions(self):
        ret = self.states()
        ret.extend(self.transitions())
        return ret

    def parameters(self):
        return self._parameter_names

    def attributes(self, state_or_trans):
        return self.by_name[state_or_trans]['attributes']

    def assess(self, model_function):
        detailed_results = {}
        model_energy_list = []
        real_energy_list = []
        model_rel_energy_list = []
        model_state_energy_list = []
        model_duration_list = []
        real_duration_list = []
        model_timeout_list = []
        real_timeout_list = []
        for name, elem in sorted(self.by_name.items()):
            detailed_results[name] = {}
            for key in elem['attributes']:
                predicted_data = np.array(list(map(lambda i: model_function(name, key, param=elem['param'][i]), range(len(elem[key])))))
                measures = regression_measures(predicted_data, elem[key])
                detailed_results[name][key] = measures

        for trace in self.traces:
            if trace['id'] not in self.ignore_trace_indexes:
                for rep_id in range(len(trace['trace'][0]['offline'])):
                    model_energy = 0.
                    real_energy = 0.
                    model_rel_energy = 0.
                    model_state_energy = 0.
                    model_duration = 0.
                    real_duration = 0.
                    model_timeout = 0.
                    real_timeout = 0.
                    for i, trace_part in enumerate(trace['trace']):
                        name = trace_part['name']
                        prev_name = trace['trace'][i-1]['name']
                        isa = trace_part['isa']
                        if name != 'UNINITIALIZED':
                            param = trace_part['offline_aggregates']['param'][rep_id]
                            prev_param = trace['trace'][i-1]['offline_aggregates']['param'][rep_id]
                            power = trace_part['offline'][rep_id]['uW_mean']
                            duration = trace_part['offline'][rep_id]['us']
                            prev_duration = trace['trace'][i-1]['offline'][rep_id]['us']
                            real_energy += power * duration
                            if isa == 'state':
                                model_energy += model_function(name, 'power', param=param) * duration
                            else:
                                model_energy += model_function(name, 'energy', param=param)
                                # If i == 1, the previous state was UNINITIALIZED, for which we do not have model data
                                if i == 1:
                                    model_rel_energy += model_function(name, 'energy', param=param)
                                else:
                                    model_rel_energy += model_function(prev_name, 'power', param=prev_param) * (prev_duration + duration)
                                    model_state_energy += model_function(prev_name, 'power', param=prev_param) * (prev_duration + duration)
                                model_rel_energy += model_function(name, 'rel_energy_prev', param=param)
                                real_duration += duration
                                model_duration += model_function(name, 'duration', param=param)
                                if 'plan' in trace_part and trace_part['plan']['level'] == 'epilogue':
                                    real_timeout += trace_part['offline'][rep_id]['timeout']
                                    model_timeout += model_function(name, 'timeout', param=param)
                    real_energy_list.append(real_energy)
                    model_energy_list.append(model_energy)
                    model_rel_energy_list.append(model_rel_energy)
                    model_state_energy_list.append(model_state_energy)
                    real_duration_list.append(real_duration)
                    model_duration_list.append(model_duration)
                    real_timeout_list.append(real_timeout)
                    model_timeout_list.append(model_timeout)

        return {
            'by_dfa_component' : detailed_results,
            'duration_by_trace' : regression_measures(np.array(model_duration_list), np.array(real_duration_list)),
            'energy_by_trace' : regression_measures(np.array(model_energy_list), np.array(real_energy_list)),
            'timeout_by_trace' : regression_measures(np.array(model_timeout_list), np.array(real_timeout_list)),
            'rel_energy_by_trace' : regression_measures(np.array(model_rel_energy_list), np.array(real_energy_list)),
            'state_energy_by_trace' : regression_measures(np.array(model_state_energy_list), np.array(real_energy_list)),
        }



class MIMOSA:
    """
    MIMOSA log loader for DFA traces with auto-calibration.

    Expects a MIMOSA log file generated via dfatool and a dfatool-generated
    benchmark: There is an automatic calibration step at the start and the
    trigger pin is high iff a transition is active. The resulting data
    is a list of state/transition/state/transition/... measurements.
    """

    def __init__(self, voltage, shunt, verbose = True):
        """
        Initialize MIMOSA loader for a specific voltage and shunt setting.

        arguments:
        voltage -- MIMOSA voltage used for measurements
        shunt -- Shunt value in Ohms
        verbose -- notify about invalid data and the likes
        """
        self.voltage = voltage
        self.shunt = shunt
        self.verbose = verbose
        self.r1 = 984 # "1k"
        self.r2 = 99013 # "100k"

    def charge_to_current_nocal(self, charge):
        u"""Convert charge per 10µs to mean currents without accounting for calibration."""
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
        """Load a MIMOSA log archive from a raw bytestring."""
        with io.BytesIO(raw_data) as data_object:
            with tarfile.open(fileobj = data_object) as tf:
                return self._load_tf(tf)

    def load_file(self, filename):
        """Load a MIMOSA log archive from a filename."""
        with tarfile.open(filename) as tf:
            return self._load_tf(tf)

    def currents_nocal(self, charges):
        u"""Convert charge per 10µs to mean currents without accounting for calibration."""
        ua_max = 1.836 / self.shunt * 1000000
        ua_step = ua_max / 65535
        return charges.astype(np.double) * ua_step

    def trigger_edges(self, triggers):
        """
        Return indexes of trigger edges (both 0->1 and 1->0) in log data.

        arguments:
        triggers -- trigger array as returned by load_data

        Ignores the first 10 seconds, which are used for calibration and may
        contain bogus triggers due to DUT resets. Returns a list of int.
        """
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
        """
        Return start/stop indexes of calibration measurements.

        arguments:
        currents -- uncalibrated currents as reported by MIMOSA. For best results,
            it may help to use a running mean, like so:
            currents = running_mean(currents_nocal(..., 10))

        Returns six indexes:
        - Disconnected start
        - Disconnected  stop
        - R1 (1 kOhm) start
        - R1 (1 kOhm) stop
        - R2 (100 kOhm) start
        - R2 (100 kOhm) stop
        """
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
        u"""
        Calculate calibration function from previously determined calibration phase.

        arguments:
        charges -- raw charges from MIMOSA
        cal_edges -- calibration edges as returned by calibration_edges

        returns (calibration_function, calibration_data):
        calibration_function -- charge in pJ (float) -> current in uA (float).
            Converts the amount of charge in a 10 µs interval to the
            mean current during the same interval.
        calibration_data -- dict containing the following keys:
            edges -- calibration points in the log file, in µs
            offset -- ...
            offset2 --  ...
            slope_low -- ...
            slope_high -- ...
            add_low -- ...
            add_high -- ..
            r0_err_uW -- mean error of uncalibrated data at "∞ Ohm" in µW
            r0_std_uW -- standard deviation of uncalibrated data at "∞ Ohm" in µW
            r1_err_uW -- mean error of uncalibrated data at 1 kOhm
            r1_std_uW -- stddev at 1 kOhm
            r2_err_uW -- mean error at 100 kOhm
            r2_std_uW -- stddev at 100 kOhm
        """
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
            vprint(self.verbose, '[W] 0 uA == %.f uA during calibration' % (ua_r2))
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

    """
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
    """

    def analyze_states(self, charges, trigidx, ua_func):
        u"""
        Split log data into states and transitions and return mean power and duration for each element.

        arguments:
        charges -- raw charges (each element describes the charge transferred during 10 µs)
        trigidx -- "charges" indexes corresponding to a trigger edge
        ua_func -- charge -> current function as returned by calibration_function

        returns a list of (alternating) states and transitions.
        Each element is a dict containing:
            - isa: 'state' oder 'transition'
            - clip_rate: range(0..1) Anteil an Clipping im Energieverbrauch
            - raw_mean: Mittelwert der Rohwerte
            - raw_std: Standardabweichung der Rohwerte
            - uW_mean: Mittelwert der (kalibrierten) Leistungsaufnahme
            - uW_std: Standardabweichung der (kalibrierten) Leistungsaufnahme
            - us: Dauer

            Nur falls isa == 'transition':
            - timeout: Dauer des vorherigen Zustands
            - uW_mean_delta_prev: Differenz zwischen uW_mean und uW_mean des vorherigen Zustands
            - uW_mean_delta_next: Differenz zwischen uW_mean und uW_mean des Folgezustands
        """
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
                    vprint(self.verbose, "ERR: duration %d vs %d" % (data['us'], ssum))

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
