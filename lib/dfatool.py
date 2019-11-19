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
import hashlib
from multiprocessing import Pool
from automata import PTA
from functions import analytic
from functions import AnalyticFunction
from parameters import ParamStats
from utils import vprint, is_numeric, soft_cast_int, param_slice_eq, remove_index_from_tuple
from utils import by_name_to_by_param, match_parameter_values

try:
    from pubcode import Code128
    import zbar
    zbar_available = True
except ImportError:
    zbar_available = False


arg_support_enabled = True

def running_mean(x: np.ndarray, N: int) -> np.ndarray:
    """
    Compute `N` elements wide running average over `x`.

    :param x: 1-Dimensional NumPy array
    :param N: how many items to average
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def gplearn_to_function(function_str: str):
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

def append_if_set(aggregate: dict, data: dict, key: str):
    """Append data[key] to aggregate if key in data."""
    if key in data:
        aggregate.append(data[key])

def mean_or_none(arr):
    """
    Compute mean of NumPy array `arr`, return -1 if empty.

    :param arr: 1-Dimensional NumPy array
    """
    if len(arr):
        return np.mean(arr)
    return -1

def aggregate_measures(aggregate: float, actual: list) -> dict:
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

def regression_measures(predicted: np.ndarray, actual: np.ndarray):
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
    #mean = np.mean(actual)
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

    def load_data(self, filename: str):
        """
        Load log data from filename, return timestamps and currents.

        Returns two one-dimensional NumPy arrays: timestamps and corresponding currents.
        """
        with open(filename) as f:
            for i, _ in enumerate(f):
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

def _xv_partition_montecarlo(length):
    shuffled = np.random.permutation(np.arange(length))
    border = int(length * float(2) / 3)
    training = shuffled[:border]
    validation = shuffled[border:]
    return (training, validation)

class CrossValidator:
    """
    Cross-Validation helper for model generation.

    Given a set of measurements and a model class, it will partition the
    data into training and validation sets, train the model on the training
    set, and assess its quality on the validation set. This is repeated
    several times depending on cross-validation algorithm and configuration.
    Reports the mean model error over all cross-validation runs.
    """

    def __init__(self, model_class, by_name, parameters, arg_count):
        """
        Create a new CrossValidator object.

        Does not perform cross-validation yet.

        arguments:
        model_class -- model class/type used for model synthesis,
            e.g. PTAModel or AnalyticModel. model_class must have a
            constructor accepting (by_name, parameters, arg_count, verbose = False)
            and provide an assess method.
        by_name -- measurements aggregated by state/transition/function/... name.
            Layout: by_name[name][attribute] = list of data. Additionally,
            by_name[name]['attributes'] must be set to the list of attributes,
            e.g. ['power'] or ['duration', 'energy'].
        """
        self.model_class = model_class
        self.by_name = by_name
        self.names = sorted(by_name.keys())
        self.parameters = sorted(parameters)
        self.arg_count = arg_count

    def montecarlo(self, model_getter, count = 200):
        """
        Perform Monte Carlo cross-validation and return average model quality.

        The by_name data is randomly divided into 2/3 training and 1/3
        validation. After creating a model for the training set, the
        model type returned by model_getter is evaluated on the validation set.
        This is repeated count times (defaulting to 200); the average of all
        measures is returned to the user.

        arguments:
        model_getter -- function with signature (model_object) -> model,
            e.g. lambda m: m.get_fitted()[0] to evaluate the parameter-aware
            model with automatic parameter detection.
        count -- number of validation runs to perform, defaults to 200

        return value:
        dict of model quality measures.
        {
            'by_name' : {
                for each name: {
                    for each attribute: {
                        'mae' : mean of all mean absolute errors
                        'mae_list' : list of the individual MAE values encountered during cross-validation
                        'smape' : mean of all symmetric mean absolute percentage errors
                        'smape_list' : list of the individual SMAPE values encountered during cross-validation
                    }
                }
            }
        }
        """
        ret = {
            'by_name' : dict()
        }

        for name in self.names:
            ret['by_name'][name] = dict()
            for attribute in self.by_name[name]['attributes']:
                ret['by_name'][name][attribute] = {
                    'mae_list': list(),
                    'smape_list': list()
                }

        for _ in range(count):
            res = self._single_montecarlo(model_getter)
            for name in self.names:
                for attribute in self.by_name[name]['attributes']:
                    ret['by_name'][name][attribute]['mae_list'].append(res['by_name'][name][attribute]['mae'])
                    ret['by_name'][name][attribute]['smape_list'].append(res['by_name'][name][attribute]['smape'])

        for name in self.names:
            for attribute in self.by_name[name]['attributes']:
                ret['by_name'][name][attribute]['mae'] = np.mean(ret['by_name'][name][attribute]['mae_list'])
                ret['by_name'][name][attribute]['smape'] = np.mean(ret['by_name'][name][attribute]['smape_list'])

        return ret

    def _single_montecarlo(self, model_getter):
        training = dict()
        validation = dict()
        for name in self.names:
            training[name] = {
                'attributes' : self.by_name[name]['attributes']
            }
            validation[name] = {
                'attributes' : self.by_name[name]['attributes']
            }

            if 'isa' in self.by_name[name]:
                training[name]['isa'] = self.by_name[name]['isa']
                validation[name]['isa'] = self.by_name[name]['isa']

            data_count = len(self.by_name[name]['param'])
            training_subset, validation_subset = _xv_partition_montecarlo(data_count)

            for attribute in self.by_name[name]['attributes']:
                self.by_name[name][attribute] = np.array(self.by_name[name][attribute])
                training[name][attribute] = self.by_name[name][attribute][training_subset]
                validation[name][attribute] = self.by_name[name][attribute][validation_subset]

            # We can't use slice syntax for 'param', which may contain strings and other odd values
            training[name]['param'] = list()
            validation[name]['param'] = list()
            for idx in training_subset:
                training[name]['param'].append(self.by_name[name]['param'][idx])
            for idx in validation_subset:
                validation[name]['param'].append(self.by_name[name]['param'][idx])

        training_data = self.model_class(training, self.parameters, self.arg_count, verbose = False)
        training_model = model_getter(training_data)
        validation_data = self.model_class(validation, self.parameters, self.arg_count, verbose = False)

        return validation_data.assess(training_model)


def _preprocess_mimosa(measurement):
    setup = measurement['setup']
    mim = MIMOSA(float(setup['mimosa_voltage']), int(setup['mimosa_shunt']))
    try:
        charges, triggers = mim.load_data(measurement['content'])
        trigidx = mim.trigger_edges(triggers)
    except EOFError as e:
        mim.errors.append('MIMOSA logfile error: {}'.format(e))
        trigidx = list()

    if len(trigidx) == 0:
        mim.errors.append('MIMOSA log has no triggers')
        return {
            'fileno' : measurement['fileno'],
            'info' : measurement['info'],
            'has_datasource_error' : len(mim.errors) > 0,
            'datasource_errors' : mim.errors,
            'expected_trace' : measurement['expected_trace'],
            'repeat_id' : measurement['repeat_id'],
        }

    cal_edges = mim.calibration_edges(running_mean(mim.currents_nocal(charges[0:trigidx[0]]), 10))
    calfunc, caldata = mim.calibration_function(charges, cal_edges)
    vcalfunc = np.vectorize(calfunc, otypes=[np.float64])

    processed_data = {
        'fileno' : measurement['fileno'],
        'info' : measurement['info'],
        'triggers' : len(trigidx),
        'first_trig' : trigidx[0] * 10,
        'calibration' : caldata,
        'energy_trace' : mim.analyze_states(charges, trigidx, vcalfunc),
        'has_datasource_error' : len(mim.errors) > 0,
        'datasource_errors' : mim.errors,
    }

    for key in ['expected_trace', 'repeat_id']:
        if key in measurement:
            processed_data[key] = measurement[key]

    return processed_data

def _preprocess_etlog(measurement):
    setup = measurement['setup']
    etlog = EnergyTraceLog(float(setup['voltage']), int(setup['state_duration']), measurement['transition_names'])
    try:
        etlog.load_data(measurement['content'])
        states_and_transitions = etlog.analyze_states(measurement['expected_trace'], measurement['repeat_id'])
    except EOFError as e:
        etlog.errors.append('EnergyTrace logfile error: {}'.format(e))
        trigidx = list()

    processed_data = {
        'fileno' : measurement['fileno'],
        'repeat_id' : measurement['repeat_id'],
        'info' : measurement['info'],
        'expected_trace' : measurement['expected_trace'],
        'energy_trace' : states_and_transitions,
        'has_datasource_error' : len(etlog.errors) > 0,
        'datasource_errors' : etlog.errors,
    }

    return processed_data

class TimingData:
    """
    Loader for timing model traces measured with on-board timers using `harness.OnboardTimerHarness`.

    Excpets a specific trace format and UART log output (as produced by
    generate-dfa-benchmark.py). Prunes states from output. (TODO)
    """

    def __init__(self, filenames):
        """
        Create a new TimingData object.

        Each filenames element corresponds to a measurement run.
        """
        self.filenames = filenames.copy()
        self.traces_by_fileno = []
        self.setup_by_fileno = []
        self.preprocessed = False
        self._parameter_names = None
        self.version = 0

    def _concatenate_analyzed_traces(self):
        self.traces = []
        for trace_group in self.traces_by_fileno:
            for trace in trace_group:
                # TimingHarness logs states, but does not aggregate any data for them at the moment -> throw all states away
                transitions = list(filter(lambda x: x['isa'] == 'transition', trace['trace']))
                self.traces.append({
                    'id' : trace['id'],
                    'trace': transitions,
                })
        for i, trace in enumerate(self.traces):
            trace['orig_id'] = trace['id']
            trace['id'] = i
            for log_entry in trace['trace']:
                paramkeys = sorted(log_entry['parameter'].keys())
                if not 'param' in log_entry['offline_aggregates']:
                    log_entry['offline_aggregates']['param'] = list()
                if 'duration' in log_entry['offline_aggregates']:
                    for i in range(len(log_entry['offline_aggregates']['duration'])):
                        paramvalues = list()
                        for paramkey in paramkeys:
                            if type(log_entry['parameter'][paramkey]) is list:
                                paramvalues.append(soft_cast_int(log_entry['parameter'][paramkey][i]))
                            else:
                                paramvalues.append(soft_cast_int(log_entry['parameter'][paramkey]))
                        if arg_support_enabled and 'args' in log_entry:
                            paramvalues.extend(map(soft_cast_int, log_entry['args']))
                        log_entry['offline_aggregates']['param'].append(paramvalues)

    def _preprocess_0(self):
        for filename in self.filenames:
            with open(filename, 'r') as f:
                log_data = json.load(f)
                self.traces_by_fileno.extend(log_data['traces'])
        self._concatenate_analyzed_traces()

    def get_preprocessed_data(self, verbose = True):
        """
        Return a list of DFA traces annotated with timing and parameter data.

        Suitable for the PTAModel constructor.
        See PTAModel(...) docstring for format details.
        """
        self.verbose = verbose
        if self.preprocessed:
            return self.traces
        if self.version == 0:
            self._preprocess_0()
        self.preprocessed = True
        return self.traces

def sanity_check_aggregate(aggregate):
    for key in aggregate:
        if not 'param' in aggregate[key]:
            raise RuntimeError('aggregate[{}][param] does not exist'.format(key))
        if not 'attributes' in aggregate[key]:
            raise RuntimeError('aggregate[{}][attributes] does not exist'.format(key))
        for attribute in aggregate[key]['attributes']:
            if not attribute in aggregate[key]:
                raise RuntimeError('aggregate[{}][{}] does not exist, even though it is contained in aggregate[{}][attributes]'.format(key, attribute, key))
            param_len = len(aggregate[key]['param'])
            attr_len = len(aggregate[key][attribute])
            if param_len != attr_len:
                raise RuntimeError('parameter mismatch: len(aggregate[{}][param]) == {} != len(aggregate[{}][{}]) == {}'.format(key, param_len, key, attribute, attr_len))


class RawData:
    """
    Loader for hardware model traces measured with MIMOSA.

    Expects a specific trace format and UART log output (as produced by the
    dfatool benchmark generator). Loads data, prunes bogus measurements, and
    provides preprocessed data suitable for PTAModel.
    """

    def __init__(self, filenames):
        """
        Create a new RawData object.

        Each filename element corresponds to a measurement run. It must be a tar archive with the following contents:

        Version 0:

        * `setup.json`: measurement setup. Must contain the keys `state_duration` (how long each state is active, in ms),
          `mimosa_voltage` (voltage applied to dut, in V), and `mimosa_shunt` (shunt value, in Ohm)
        * `src/apps/DriverEval/DriverLog.json`: PTA traces and parameters for this benchmark.
          Layout: List of traces, each trace has an 'id' (numeric, starting with 1) and 'trace' (list of states and transitions) element.
          Each trace has an even number of elements, starting with the first state (usually `UNINITIALIZED`) and ending with a transition.
          Each state/transition must have the members `.parameter` (parameter values, empty string or None if unknown), `.isa` ("state" or "transition") and `.name`.
          Each transition must additionally contain `.plan.level` ("user" or "epilogue").
          Example: `[ {"id": 1, "trace": [ {"parameter": {...}, "isa": "state", "name": "UNINITIALIZED"}, ...] }, ... ]
        * At least one `*.mim` file. Each file corresponds to a single execution of the entire benchmark (i.e., all runs described in DriverLog.json) and starts with a MIMOSA Autocal calibration sequence.
          MIMOSA files are parsed by the `MIMOSA` class.

        Version 1:

        * `ptalog.json`: measurement setup and traces. Contents:
          `.opt.sleep`: state duration
          `.opt.pta`: PTA
          `.opt.traces`: list of sub-benchmark traces (the benchmark may have been split due to code size limitations). Each item is a list of traces as returned by `harness.traces`:
            `.opt.traces[]`: List of traces. Each trace has an 'id' (numeric, starting with 1) and 'trace' (list of states and transitions) element.
              Each state/transition must have the members '`parameter` (dict with normalized parameter values), `.isa` ("state" or "transition") and `.name`
              Each transition must additionally contain `.args`
          `.opt.files`: list of coresponding MIMOSA measurements.
            `.opt.files[]` = ['abc123.mim', ...]
          `.opt.configs`: ....
        * MIMOSA log files (`*.mim`) as specified in `.opt.files`

        Version 2:

        * `ptalog.json`: measurement setup and traces. Contents:
          `.opt.sleep`: state duration
          `.opt.pta`: PTA
          `.opt.traces`: list of sub-benchmark traces (the benchmark may have been split due to code size limitations). Each item is a list of traces as returned by `harness.traces`:
            `.opt.traces[]`: List of traces. Each trace has an 'id' (numeric, starting with 1) and 'trace' (list of states and transitions) element.
              Each state/transition must have the members '`parameter` (dict with normalized parameter values), `.isa` ("state" or "transition") and `.name`
              Each transition must additionally contain `.args` and `.duration`
              * `.duration`: list of durations, one per repetition
          `.opt.files`: list of coresponding EnergyTrace measurements.
            `.opt.files[]` = ['abc123.etlog', ...]
          `.opt.configs`: ....
        * EnergyTrace log files (`*.etlog`) as specified in `.opt.files`

        tbd
        """
        self.filenames = filenames.copy()
        self.traces_by_fileno = []
        self.setup_by_fileno = []
        self.version = 0
        self.preprocessed = False
        self._parameter_names = None
        self.ignore_clipping = False

        with tarfile.open(filenames[0]) as tf:
            for member in tf.getmembers():
                if member.name == 'ptalog.json':
                    self.version = 1
                    # might also be version 2
                    # depends on whether *.etlog exists or not
                elif '.etlog' in member.name:
                    self.version = 2
                    break

        self.set_cache_file()
        self.load_cache()

    def set_cache_file(self):
        cache_key = hashlib.sha256('!'.join(self.filenames).encode()).hexdigest()
        self.cache_dir = os.path.dirname(self.filenames[0]) + '/cache'
        self.cache_file = '{}/{}.json'.format(self.cache_dir, cache_key)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                self.traces = cache_data['traces']
                self.preprocessing_stats = cache_data['preprocessing_stats']
                self.preprocessed = True

    def save_cache(self):
        try:
            os.mkdir(self.cache_dir)
        except FileExistsError:
            pass
        with open(self.cache_file, 'w') as f:
            cache_data = {
                'traces' : self.traces,
                'preprocessing_stats' : self.preprocessing_stats
            }
            json.dump(cache_data, f)

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

    def _measurement_is_valid_2(self, processed_data):
        """
        Check if a dfatool v2 measurement is valid.

        processed_data layout:
        'fileno' : measurement['fileno'],
        'info' : measurement['info'],
        'energy_trace' : etlog.analyze_states()
            A sequence of unnamed, unparameterized states and transitions with
            power and timing data
        'expected_trace' : trace from PTA DFS (with parameter data)
        etlog.analyze_states returns a list of (alternating) states and transitions.
        Each element is a dict containing:
            - isa: 'state' oder 'transition'
            - W_mean: Mittelwert der (kalibrierten) Leistungsaufnahme
            - W_std: Standardabweichung der (kalibrierten) Leistungsaufnahme
            - s: duration

            if isa == 'transition':
            - W_mean_delta_prev: Differenz zwischen W_mean und W_mean des vorherigen Zustands
            - W_mean_delta_next: Differenz zwischen W_mean und W_mean des Folgezustands
        """
        setup = self.setup_by_fileno[processed_data['fileno']]
        traces = processed_data['expected_trace']

        # Check for low-level parser errors
        if processed_data['has_datasource_error']:
            processed_data['error'] = '; '.join(processed_data['datasource_errors'])
            return False

        # Note that the low-level parser (EnergyTraceLog) already checks
        # whether the transition count is correct

        return True

    def _measurement_is_valid_01(self, processed_data):
        """
        Check if a dfatool v0 or v1 measurement is valid.

        processed_data layout:
        'fileno' : measurement['fileno'],
        'info' : measurement['info'],
        'triggers' : len(trigidx),
        'first_trig' : trigidx[0] * 10,
        'calibration' : caldata,
        'energy_trace' : mim.analyze_states(charges, trigidx, vcalfunc)
            A sequence of unnamed, unparameterized states and transitions with
            power and timing data
        'expected_trace' : trace from PTA DFS (with parameter data)
        mim.analyze_states returns a list of (alternating) states and transitions.
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
        setup = self.setup_by_fileno[processed_data['fileno']]
        if 'expected_trace' in processed_data:
            traces = processed_data['expected_trace']
        else:
            traces = self.traces_by_fileno[processed_data['fileno']]
        state_duration = setup['state_duration']

        # Check MIMOSA error
        if processed_data['has_datasource_error']:
            processed_data['error'] = '; '.join(processed_data['datasource_errors'])
            return False

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
            offline_trace_part = processed_data['energy_trace'][offline_idx]
            online_trace_part = traces[online_run_idx]['trace'][online_trace_part_idx]

            if self._parameter_names == None:
                self._parameter_names = sorted(online_trace_part['parameter'].keys())

            if sorted(online_trace_part['parameter'].keys()) != self._parameter_names:
                processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) has inconsistent parameter set: should be {param_want:s}, is {param_is:s}'.format(
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
            if offline_idx != 0 and offline_trace_part['clip_rate'] != 0 and not self.ignore_clipping:
                processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) was clipping {clip:f}% of the time'.format(
                    off_idx = offline_idx, on_idx = online_run_idx,
                    on_sub = online_trace_part_idx,
                    on_name = online_trace_part['name'],
                    clip = offline_trace_part['clip_rate'] * 100,
                )
                return False


            if online_trace_part['isa'] == 'state' and online_trace_part['name'] != 'UNINITIALIZED' and len(traces[online_run_idx]['trace']) > online_trace_part_idx+1:
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

    def _merge_online_and_offline(self, measurement):
        # Edits self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline']
        # and self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline_aggregates'] in place
        # (appends data from measurement['energy_trace'])
        # If measurement['expected_trace'] exists, it is edited in place instead
        online_datapoints = []
        if 'expected_trace' in measurement:
            traces = measurement['expected_trace']
            traces = self.traces_by_fileno[measurement['fileno']]
        else:
            traces = self.traces_by_fileno[measurement['fileno']]
        for run_idx, run in enumerate(traces):
            for trace_part_idx in range(len(run['trace'])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, online_ref in enumerate(online_datapoints):
            online_run_idx, online_trace_part_idx = online_ref
            offline_trace_part = measurement['energy_trace'][offline_idx]
            online_trace_part = traces[online_run_idx]['trace'][online_trace_part_idx]

            if not 'offline' in online_trace_part:
                online_trace_part['offline'] = [offline_trace_part]
            else:
                online_trace_part['offline'].append(offline_trace_part)

            paramkeys = sorted(online_trace_part['parameter'].keys())

            paramvalues = list()

            for paramkey in paramkeys:
                if type(online_trace_part['parameter'][paramkey]) is list:
                    paramvalues.append(soft_cast_int(online_trace_part['parameter'][paramkey][measurement['repeat_id']]))
                else:
                    paramvalues.append(soft_cast_int(online_trace_part['parameter'][paramkey]))

            # NB: Unscheduled transitions do not have an 'args' field set.
            # However, they should only be caused by interrupts, and
            # interrupts don't have args anyways.
            if arg_support_enabled and 'args' in online_trace_part:
                paramvalues.extend(map(soft_cast_int, online_trace_part['args']))

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
            online_trace_part['offline_aggregates']['param'].append(paramvalues)
            if online_trace_part['isa'] == 'transition':
                online_trace_part['offline_aggregates']['rel_energy_prev'].append(
                    offline_trace_part['uW_mean_delta_prev'] * (offline_trace_part['us'] - 20))
                online_trace_part['offline_aggregates']['rel_energy_next'].append(
                    offline_trace_part['uW_mean_delta_next'] * (offline_trace_part['us'] - 20))
                online_trace_part['offline_aggregates']['timeout'].append(
                    offline_trace_part['timeout'])

    def _merge_online_and_etlog(self, measurement):
        # Edits self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline']
        # and self.traces_by_fileno[measurement['fileno']][*]['trace'][*]['offline_aggregates'] in place
        # (appends data from measurement['energy_trace'])
        online_datapoints = []
        traces = self.traces_by_fileno[measurement['fileno']]
        for run_idx, run in enumerate(traces):
            for trace_part_idx in range(len(run['trace'])):
                online_datapoints.append((run_idx, trace_part_idx))
        for offline_idx, online_ref in enumerate(online_datapoints):
            online_run_idx, online_trace_part_idx = online_ref
            offline_trace_part = measurement['energy_trace'][offline_idx]
            online_trace_part = traces[online_run_idx]['trace'][online_trace_part_idx]

            if not 'offline' in online_trace_part:
                online_trace_part['offline'] = [offline_trace_part]
            else:
                online_trace_part['offline'].append(offline_trace_part)

            paramkeys = sorted(online_trace_part['parameter'].keys())

            paramvalues = list()

            for paramkey in paramkeys:
                if type(online_trace_part['parameter'][paramkey]) is list:
                    paramvalues.append(soft_cast_int(online_trace_part['parameter'][paramkey][measurement['repeat_id']]))
                else:
                    paramvalues.append(soft_cast_int(online_trace_part['parameter'][paramkey]))

            # NB: Unscheduled transitions do not have an 'args' field set.
            # However, they should only be caused by interrupts, and
            # interrupts don't have args anyways.
            if arg_support_enabled and 'args' in online_trace_part:
                paramvalues.extend(map(soft_cast_int, online_trace_part['args']))

            # only if isa == 'state'
            if 'offline_aggregates' not in online_trace_part:
                online_trace_part['offline_aggregates'] = {
                    'duration' : list()
                }

            offline_aggregates = online_trace_part['offline_aggregates']

            if not 'power' in offline_aggregates:
                online_trace_part['offline_attributes'] = ['power', 'duration', 'energy']
                offline_aggregates['power'] = list()
                offline_aggregates['power_std'] = list()
                offline_aggregates['energy'] = list()
                offline_aggregates['paramkeys'] = list()
                offline_aggregates['param'] = list()

                #if online_trace_part['isa'] == 'transitions':
                #    online_trace_part['offline_attributes'].extend(['rel_energy_prev', 'rel_energy_next'])
                #    offline_aggregates['rel_energy_prev'] = list()
                #    offline_aggregates['rel_energy_next'] = list()

            offline_aggregates['power'].append(offline_trace_part['W_mean'] * 1e6)
            offline_aggregates['power_std'].append(offline_trace_part['W_std'] * 1e6)
            offline_aggregates['energy'].append(offline_trace_part['W_mean'] * offline_trace_part['s'] * 1e12)
            offline_aggregates['paramkeys'].append(paramkeys)
            offline_aggregates['param'].append(paramvalues)

            if online_trace_part['isa'] == 'state':
                offline_aggregates['duration'].append(offline_trace_part['s'] * 1e6)

            #if online_trace_part['isa'] == 'transition':
            #    offline_aggregates['rel_energy_prev'].append(offline_trace_part['W_mean_delta_prev'] * offline_trace_part['s'] * 1e12)
            #    offline_aggregates['rel_energy_next'].append(offline_trace_part['W_mean_delta_next'] * offline_trace_part['s'] * 1e12)


    def _concatenate_traces(self, list_of_traces):
        trace_output = list()
        for trace in list_of_traces:
            trace_output.extend(trace.copy())
        for i, trace in enumerate(trace_output):
            trace['orig_id'] = trace['id']
            trace['id'] = i
        return trace_output

    def get_preprocessed_data(self, verbose = True):
        """
        Return a list of DFA traces annotated with energy, timing, and parameter data.

        Each DFA trace contains the following elements:
         * `id`: Numeric ID, starting with 1
         * `total_energy`: Total amount of energy (as measured by MIMOSA) in the entire trace
         * `orig_id`: Original trace ID. May differ when concatenating multiple (different) benchmarks into one analysis, i.e., when calling RawData() with more than one file argument.
         * `trace`: List of the individual states and transitions in this trace. Always contains an even number of elements, staring with the first state (typically "UNINITIALIZED") and ending with a transition.

        Each trace element (that is, an entry of the `trace` list mentioned above) contains the following elements:
         * `isa`: "state" or "transition"
         * `name`: name
         * `offline`: List of offline measumerents for this state/transition. Each entry contains a result for this state/transition during one benchmark execution.
           Entry contents:
            - `clip_rate`: rate of clipped energy measurements, 0 .. 1
            - `raw_mean`: mean raw MIMOSA value
            - `raw_std`: standard deviation of raw MIMOSA value
            - `uW_mean`: mean power draw, uW
            - `uw_std`: standard deviation of power draw, uW
            - `us`: state/transition duration, us
            - `uW_mean_delta_prev`: (only for transitions) difference between uW_mean of this transition and uW_mean of previous state
            - `uW_mean_elta_next`: (only for transitions) difference between uW_mean of this transition and uW_mean of next state
            - `timeout`: (only for transitions) duration of previous state, us
         * `offline_aggregates`: Aggregate of `offline` entries. dict of lists, each list entry has the same length
            - `duration`: state/transition durations ("us"), us
            - `energy`: state/transition energy ("us * uW_mean"), us
            - `power`: mean power draw ("uW_mean"), uW
            - `power_std`: standard deviations of power draw ("uW_std"), uW^2
            - `paramkeys`: List of lists, each sub-list contains the parameter names corresponding to the `param` entries
            - `param`: List of lists, each sub-list contains the parameter values for this measurement. Typically, all sub-lists are the same.
            - `rel_energy_prev`: (only for transitions) transition energy relative to previous state mean power, pJ
            - `rel_energy_next`: (only for transitions) transition energy relative to next state mean power, pJ
            - `timeout`: (only for transitions) duration of previous state, us
         * `offline_attributes`: List containing the keys of `offline_aggregates` which are meant to be part of themodel.
           This list ultimately decides which hardware/software attributes the model describes.
           If isa == state, it contains power, duration, energy
           If isa == transition, it contains power, duration, energy, rel_energy_prev, rel_energy_next, timeout
         * `online`: List of online estimations for this state/transition. Each entry contains a result for this state/transition during one benchmark execution.
          Entry contents for isa == state:
            - `time`: state/transition 
          Entry contents for isa == transition:
            - `timeout`: Duration of previous state, measured using on-board timers
         * `parameter`: dictionary describing parameter values for this state/transition. Parameter values refer to the begin of the state/transition and do not account for changes made by the transition.
         * `plan`: Dictionary describing expected behaviour according to schedule / offline model.
           Contents for isa == state: `energy`, `power`, `time`
           Contents for isa == transition: `energy`, `timeout`, `level`.
           If level is "user", the transition is part of the regular driver API. If level is "epilogue", it is an interrupt service routine and not called explicitly.
        Each transition also contains:
         * `args`: List of arguments the corresponding function call was called with. args entries are strings which are not necessarily numeric
         * `code`: List of function name (first entry) and arguments (remaining entries) of the corresponding function call
        """
        self.verbose = verbose
        if self.preprocessed:
            return self.traces
        if self.version == 0:
            self._preprocess_012(0)
        elif self.version == 1:
            self._preprocess_012(1)
        elif self.version == 2:
            self._preprocess_012(2)
        self.preprocessed = True
        self.save_cache()
        return self.traces

    def _preprocess_012(self, version):
        """Load raw MIMOSA data and turn it into measurements which are ready to be analyzed."""
        offline_data = []
        for i, filename in enumerate(self.filenames):

            if version == 0:

                with tarfile.open(filename) as tf:
                    self.setup_by_fileno.append(json.load(tf.extractfile('setup.json')))
                    self.traces_by_fileno.append(json.load(tf.extractfile('src/apps/DriverEval/DriverLog.json')))
                    for member in tf.getmembers():
                        _, extension = os.path.splitext(member.name)
                        if extension == '.mim':
                            offline_data.append({
                                'content' : tf.extractfile(member).read(),
                                'fileno' : i,
                                'info' : member,
                                'setup' : self.setup_by_fileno[i],
                            })

            elif version == 1:

                new_filenames = list()
                with tarfile.open(filename) as tf:
                    ptalog = json.load(tf.extractfile(tf.getmember('ptalog.json')))

                    # Benchmark code may be too large to be executed in a single
                    # run, so benchmarks (a benchmark is basically a list of DFA runs)
                    # may be split up. To accomodate this, ptalog['traces'] is
                    # a list of lists: ptalog['traces'][0] corresponds to the
                    # first benchmark part, ptalog['traces'][1] to the
                    # second, and so on. ptalog['traces'][0][0] is the first
                    # trace (a sequence of states and transitions) in the
                    # first benchmark part, ptalog['traces'][0][1] the second, etc.
                    #
                    # As traces are typically repeated to minimize the effect
                    # of random noise, observations for each benchmark part
                    # are also lists. In this case, this applies in two
                    # cases: traces[i][j]['parameter'][some_param] is either
                    # a value (if the parameter is controlld by software)
                    # or a list (if the parameter is known a posteriori, e.g.
                    # "how many retransmissions did this packet take?").
                    #
                    # The second case is the MIMOSA energy measurements, which
                    # are listed in ptalog['files']. ptalog['files'][0]
                    # contains a list of files for the first benchmark part,
                    # ptalog['files'][0][0] is its first iteration/repetition,
                    # ptalog['files'][0][1] the second, etc.

                    for j, traces in enumerate(ptalog['traces']):
                        new_filenames.append('{}#{}'.format(filename, j))
                        self.traces_by_fileno.append(traces)
                        self.setup_by_fileno.append({
                            'mimosa_voltage' : ptalog['configs'][j]['voltage'],
                            'mimosa_shunt' : ptalog['configs'][j]['shunt'],
                            'state_duration' : ptalog['opt']['sleep'],
                        })
                        for repeat_id, mim_file in enumerate(ptalog['files'][j]):
                            member = tf.getmember(mim_file)
                            offline_data.append({
                                'content' : tf.extractfile(member).read(),
                                'fileno' : j,
                                'info' : member,
                                'setup' : self.setup_by_fileno[j],
                                'repeat_id' : repeat_id,
                                'expected_trace' : ptalog['traces'][j],
                            })
                self.filenames = new_filenames

            elif version == 2:

                new_filenames = list()
                with tarfile.open(filename) as tf:
                    ptalog = json.load(tf.extractfile(tf.getmember('ptalog.json')))

                    # Benchmark code may be too large to be executed in a single
                    # run, so benchmarks (a benchmark is basically a list of DFA runs)
                    # may be split up. To accomodate this, ptalog['traces'] is
                    # a list of lists: ptalog['traces'][0] corresponds to the
                    # first benchmark part, ptalog['traces'][1] to the
                    # second, and so on. ptalog['traces'][0][0] is the first
                    # trace (a sequence of states and transitions) in the
                    # first benchmark part, ptalog['traces'][0][1] the second, etc.
                    #
                    # As traces are typically repeated to minimize the effect
                    # of random noise, observations for each benchmark part
                    # are also lists. In this case, this applies in two
                    # cases: traces[i][j]['parameter'][some_param] is either
                    # a value (if the parameter is controlld by software)
                    # or a list (if the parameter is known a posteriori, e.g.
                    # "how many retransmissions did this packet take?").
                    #
                    # The second case is the MIMOSA energy measurements, which
                    # are listed in ptalog['files']. ptalog['files'][0]
                    # contains a list of files for the first benchmark part,
                    # ptalog['files'][0][0] is its first iteration/repetition,
                    # ptalog['files'][0][1] the second, etc.

                    for j, traces in enumerate(ptalog['traces']):
                        new_filenames.append('{}#{}'.format(filename, j))
                        self.traces_by_fileno.append(traces)
                        self.setup_by_fileno.append({
                            'voltage' : ptalog['configs'][j]['voltage'],
                            'state_duration' : ptalog['opt']['sleep'],
                        })
                        for repeat_id, etlog_file in enumerate(ptalog['files'][j]):
                            member = tf.getmember(etlog_file)
                            offline_data.append({
                                'content' : tf.extractfile(member).read(),
                                'fileno' : j,
                                'info' : member,
                                'setup' : self.setup_by_fileno[j],
                                'repeat_id' : repeat_id,
                                'expected_trace' : ptalog['traces'][j],
                                'transition_names' : list(map(lambda x: x['name'], ptalog['pta']['transitions']))
                            })
                self.filenames = new_filenames

        with Pool() as pool:
            if self.version <= 1:
                measurements = pool.map(_preprocess_mimosa, offline_data)
            elif self.version == 2:
                measurements = pool.map(_preprocess_etlog, offline_data)

        num_valid = 0
        valid_traces = list()
        for measurement in measurements:

            if not 'energy_trace' in measurement:
                vprint(self.verbose, '[W] Skipping {ar:s}/{m:s}: {e:s}'.format(
                    ar = self.filenames[measurement['fileno']],
                    m = measurement['info'].name,
                    e = '; '.join(measurement['datasource_errors'])))
                continue

            if version == 0:
                # Strip the last state (it is not part of the scheduled measurement)
                measurement['energy_trace'].pop()
                repeat = 0
            elif version == 1:
                # The first online measurement is the UNINITIALIZED state. In v1,
                # it is not part of the expected PTA trace -> remove it.
                measurement['energy_trace'].pop(0)
                repeat = ptalog['opt']['repeat']
            elif version == 2:
                repeat = ptalog['opt']['repeat']

            if version == 0 or version == 1:
                if self._measurement_is_valid_01(measurement):
                    self._merge_online_and_offline(measurement)
                    num_valid += 1
                else:
                    vprint(self.verbose, '[W] Skipping {ar:s}/{m:s}: {e:s}'.format(
                        ar = self.filenames[measurement['fileno']],
                        m = measurement['info'].name,
                        e = measurement['error']))
            elif version == 2:
                if self._measurement_is_valid_2(measurement):
                    self._merge_online_and_etlog(measurement)
                    num_valid += 1
                else:
                    vprint(self.verbose, '[W] Skipping {ar:s}/{m:s}: {e:s}'.format(
                        ar = self.filenames[measurement['fileno']],
                        m = measurement['info'].name,
                        e = measurement['error']))
        vprint(self.verbose, '[I] {num_valid:d}/{num_total:d} measurements are valid'.format(
            num_valid = num_valid,
            num_total = len(measurements)))
        if version == 0:
            self.traces = self._concatenate_traces(self.traces_by_fileno)
        elif version == 1:
            self.traces = self._concatenate_traces(map(lambda x: x['expected_trace'], measurements))
            self.traces = self._concatenate_traces(self.traces_by_fileno)
        elif version == 2:
            self.traces = self._concatenate_traces(self.traces_by_fileno)
        self.preprocessing_stats = {
            'num_runs' : len(measurements),
            'num_valid' : num_valid
        }

class ParallelParamFit:
    """
    Fit a set of functions on parameterized measurements.

    One parameter is variale, all others are fixed. Reports the best-fitting
    function type for each parameter.
    """

    def __init__(self, by_param):
        """Create a new ParallelParamFit object."""
        self.fit_queue = []
        self.by_param = by_param

    def enqueue(self, state_or_tran, attribute, param_index, param_name, safe_functions_enabled = False, param_filter = None):
        """
        Add state_or_tran/attribute/param_name to fit queue.

        This causes fit() to compute the best-fitting function for this model part.
        """
        self.fit_queue.append({
            'key' : [state_or_tran, attribute, param_name, param_filter],
            'args' : [self.by_param, state_or_tran, attribute, param_index, safe_functions_enabled, param_filter]
        })

    def fit(self):
        """
        Fit functions on previously enqueue data.

        Fitting is one in parallel with one process per core.

        Results can be accessed using the public ParallelParamFit.results object.
        """
        with Pool() as pool:
            self.results = pool.map(_try_fits_parallel, self.fit_queue)

def _try_fits_parallel(arg):
    """
    Call _try_fits(*arg['args']) and return arg['key'] and the _try_fits result.

    Must be a global function as it is called from a multiprocessing Pool.
    """
    return {
        'key' : arg['key'],
        'result' : _try_fits(*arg['args'])
    }

def _try_fits(by_param, state_or_tran, model_attribute, param_index, safe_functions_enabled = False, param_filter: dict = None):
    """
    Determine goodness-of-fit for prediction of `by_param[(state_or_tran, *)][model_attribute]` dependence on `param_index` using various functions.

    This is done by varying `param_index` while keeping all other parameters constant and doing one least squares optimization for each function and for each combination of the remaining parameters.
    The value of the parameter corresponding to `param_index` (e.g. txpower or packet length) is the sole input to the model function.
    Only numeric parameter values (as determined by `utils.is_numeric`) are used for fitting, non-numeric values such as None or enum strings are ignored.
    Fitting is only performed if at least three distinct parameter values exist in `by_param[(state_or_tran, *)]`.

    :returns:  a dictionary with the following elements:
        best -- name of the best-fitting function (see `analytic.functions`). `None` in case of insufficient data.
        best_rmsd -- mean Root Mean Square Deviation of best-fitting function over all combinations of the remaining parameters
        mean_rmsd -- mean Root Mean Square Deviation of a reference model using the mean of its respective input data as model value
        median_rmsd -- mean Root Mean Square Deviation of a reference model using the median of its respective input data as model value
        results -- mean goodness-of-fit measures for the individual functions. See `analytic.functions` for keys and `aggregate_measures` for values

    :param by_param: measurements partitioned by state/transition/... name and parameter values.
    Example: `{('foo', (0, 2)): {'bar': [2]}, ('foo', (0, 4)): {'bar': [4]}, ('foo', (0, 6)): {'bar': [6]}}`

    :param state_or_tran: state/transition/... name for which goodness-of-fit will be calculated (first element of by_param key tuple).
    Example: `'foo'`

    :param model_attribute: attribute for which goodness-of-fit will be calculated.
    Example: `'bar'`
    
    :param param_index: index of the parameter used as model input
    :param safe_functions_enabled: Include "safe" variants of functions with limited argument range.
    :param param_filter: Only use measurements whose parameters match param_filter for fitting.
    """

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

    raw_results = dict()
    raw_results_by_param = dict()
    ref_results = {
        'mean' : list(),
        'median' : list()
    }
    results = dict()
    results_by_param = dict()

    seen_parameter_combinations = set()

    # for each parameter combination:
    for param_key in filter(lambda x: x[0] == state_or_tran and remove_index_from_tuple(x[1], param_index) not in seen_parameter_combinations and len(by_param[x]['param']) and match_parameter_values(by_param[x]['param'][0], param_filter), by_param.keys()):
        X = []
        Y = []
        num_valid = 0
        num_total = 0

        # Ensure that each parameter combination is only optimized once. Otherwise, with parameters (1, 2, 5), (1, 3, 5), (1, 4, 5) and param_index == 1,
        # the parameter combination (1, *, 5) would be optimized three times, both wasting time and biasing results towards more frequently occuring combinations of non-param_index parameters
        seen_parameter_combinations.add(remove_index_from_tuple(param_key[1], param_index))

        # for each value of the parameter denoted by param_index (all other parameters remain the same):
        for k, v in filter(lambda kv: param_slice_eq(kv[0], param_key, param_index), by_param.items()):
            num_total += 1
            if is_numeric(k[1][param_index]):
                num_valid += 1
                X.extend([float(k[1][param_index])] * len(v[model_attribute]))
                Y.extend(v[model_attribute])

        if num_valid > 2:
            X = np.array(X)
            Y = np.array(Y)
            other_parameters = remove_index_from_tuple(k[1], param_index)
            raw_results_by_param[other_parameters] = dict()
            results_by_param[other_parameters] = dict()
            for function_name, param_function in functions.items():
                if not function_name in raw_results:
                    raw_results[function_name] = dict()
                error_function = param_function.error_function
                res = optimize.least_squares(error_function, [0, 1], args=(X, Y), xtol=2e-15)
                measures = regression_measures(param_function.eval(res.x, X), Y)
                raw_results_by_param[other_parameters][function_name] = measures
                for measure, error_rate in measures.items():
                    if not measure in raw_results[function_name]:
                        raw_results[function_name][measure] = list()
                    raw_results[function_name][measure].append(error_rate)
                #print(function_name, res, measures)
            mean_measures = aggregate_measures(np.mean(Y), Y)
            ref_results['mean'].append(mean_measures['rmsd'])
            raw_results_by_param[other_parameters]['mean'] = mean_measures
            median_measures = aggregate_measures(np.median(Y), Y)
            ref_results['median'].append(median_measures['rmsd'])
            raw_results_by_param[other_parameters]['median'] = median_measures

    if not len(ref_results['mean']):
        # Insufficient data for fitting
        #print('[W] Insufficient data for fitting {}/{}/{}'.format(state_or_tran, model_attribute, param_index))
        return {
            'best' : None,
            'best_rmsd' : np.inf,
            'results' : results
        }
    
    for other_parameter_combination, other_parameter_results in raw_results_by_param.items():
        best_fit_val = np.inf
        best_fit_name = None
        results = dict()
        for function_name, result in other_parameter_results.items():
            if len(result) > 0:
                results[function_name] = result
                rmsd = result['rmsd']
                if rmsd < best_fit_val:
                    best_fit_val = rmsd
                    best_fit_name = function_name
        results_by_param[other_parameter_combination] = {
            'best': best_fit_name,
            'best_rmsd': best_fit_val,
            'mean_rmsd' : results['mean']['rmsd'],
            'median_rmsd' : results['median']['rmsd'],
            'results' : results
        }

    best_fit_val = np.inf
    best_fit_name = None
    results = dict()
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
        'results' : results,
        'results_by_other_param' : results_by_param
    }

def _num_args_from_by_name(by_name):
    num_args = dict()
    for key, value in by_name.items():
        if 'args' in value:
            num_args[key] = len(value['args'][0])
    return num_args

def get_fit_result(results, name, attribute, verbose = False, param_filter: dict = None):
    """
    Parse and sanitize fit results for state/transition/... 'name' and model attribute 'attribute'.

    Filters out results where the best function is worse (or not much better than) static mean/median estimates.

    :param results: fit results as returned by `paramfit.results`
    :param name: state/transition/... name, e.g. 'TX'
    :param attribute: model attribute, e.g. 'duration'
    :param verbose: print debug message to stdout when deliberately not using a determined fit function
    :param param_filter: 
    :returns: dict with fit result (see `_try_fits`) for each successfully fitted parameter. E.g. {'param 1': {'best' : 'function name', ...} }
    """
    fit_result = dict()
    for result in results:
        if result['key'][0] == name and result['key'][1] == attribute and result['key'][3] == param_filter and result['result']['best'] != None: # dürfte an ['best'] != None liegen-> Fit für gefilterten Kram schlägt fehl?
            this_result = result['result']
            if this_result['best_rmsd'] >= min(this_result['mean_rmsd'], this_result['median_rmsd']):
                vprint(verbose, '[I] Not modeling {} {} as function of {}: best ({:.0f}) is worse than ref ({:.0f}, {:.0f})'.format(
                    name, attribute, result['key'][2], this_result['best_rmsd'],
                    this_result['mean_rmsd'], this_result['median_rmsd']))
            # See notes on depends_on_param
            elif this_result['best_rmsd'] >= 0.8 * min(this_result['mean_rmsd'], this_result['median_rmsd']):
                vprint(verbose, '[I] Not modeling {} {} as function of {}: best ({:.0f}) is not much better than ref ({:.0f}, {:.0f})'.format(
                    name, attribute, result['key'][2], this_result['best_rmsd'],
                    this_result['mean_rmsd'], this_result['median_rmsd']))
            else:
                fit_result[result['key'][2]] = this_result
    return fit_result

class AnalyticModel:
    u"""
    Parameter-aware analytic energy/data size/... model.

    Supports both static and parameter-based model attributes, and automatic detection of parameter-dependence.

    These provide measurements aggregated by (function/state/...) name
    and (for by_param) parameter values. Layout:
    dictionary with one key per name ('send', 'TX', ...) or
    one key per name and parameter combination
    (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).

    Parameter values must be ordered corresponding to the lexically sorted parameter names.

    Each element is in turn a dict with the following elements:
    - param: list of parameter values in each measurement (-> list of lists)
    - attributes: list of keys that should be analyzed,
        e.g. ['power', 'duration']
    - for each attribute mentioned in 'attributes': A list with measurements.
      All list except for 'attributes' must have the same length.

    For example:
    parameters = ['foo_count', 'irrelevant']
    by_name = {
        'foo' : [1, 1, 2],
        'bar' : [5, 6, 7],
        'attributes' : ['foo', 'bar'],
        'param' : [[1, 0], [1, 0], [2, 0]]
    }

    methods:
    get_static -- return static (parameter-unaware) model.
    get_param_lut -- return parameter-aware look-up-table model. Cannot model parameter combinations not present in by_param.
    get_fitted -- return parameter-aware model using fitted functions for behaviour prediction.

    variables:
    names -- function/state/... names (i.e., the keys of by_name)
    parameters -- parameter names
    stats -- ParamStats object providing parameter-dependency statistics for each name and attribute
    assess -- calculate model quality
    """

    def __init__(self, by_name, parameters, arg_count = None, function_override = dict(), verbose = True, use_corrcoef = False):
        """
        Create a new AnalyticModel and compute parameter statistics.
        
        :param by_name: measurements aggregated by (function/state/...) name.
            Layout: dictionary with one key per name ('send', 'TX', ...) or
            one key per name and parameter combination
            (('send', (1, 2)), ('send', (2, 3)), ('TX', (1, 2)), ('TX', (2, 3)), ...).

            Parameter values must be ordered corresponding to the lexically sorted parameter names.

            Each element is in turn a dict with the following elements:
            - param: list of parameter values in each measurement (-> list of lists)
            - attributes: list of keys that should be analyzed,
                e.g. ['power', 'duration']
            - for each attribute mentioned in 'attributes': A list with measurements.
            All list except for 'attributes' must have the same length.

            For example:
            parameters = ['foo_count', 'irrelevant']
            by_name = {
                'foo' : [1, 1, 2],
                'duration' : [5, 6, 7],
                'attributes' : ['foo', 'duration'],
                'param' : [[1, 0], [1, 0], [2, 0]]
                # foo_count-^  ^-irrelevant
            }
        :param parameters: List of parameter names
        :param function_override: dict of overrides for automatic parameter function generation.
            If (state or transition name, model attribute) is present in function_override,
            the corresponding text string is the function used for analytic (parameter-aware/fitted)
            modeling of this attribute. It is passed to AnalyticFunction, see
            there for the required format. Note that this happens regardless of
            parameter dependency detection: The provided analytic function will be assigned
            even if it seems like the model attribute is static / parameter-independent.
        :param verbose: Print debug/info output while generating the model?
        :param use_corrcoef: use correlation coefficient instead of stddev comparison to detect whether a model attribute depends on a parameter
        """
        self.cache = dict()
        self.by_name = by_name
        self.by_param = by_name_to_by_param(by_name)
        self.names = sorted(by_name.keys())
        self.parameters = sorted(parameters)
        self.function_override = function_override.copy()
        self.verbose = verbose
        self._use_corrcoef = use_corrcoef
        self._num_args = arg_count
        if self._num_args is None:
            self._num_args = _num_args_from_by_name(by_name)

        self.stats = ParamStats(self.by_name, self.by_param, self.parameters, self._num_args, verbose = verbose, use_corrcoef = use_corrcoef)

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

    def param_index(self, param_name):
        if param_name in self.parameters:
            return self.parameters.index(param_name)
        return len(self.parameters) + int(param_name)

    def param_name(self, param_index):
        if param_index < len(self.parameters):
            return self.parameters[param_index]
        return str(param_index)

    def get_static(self, use_mean=False):
        """
        Get static model function: name, attribute -> model value.

        Uses the median of by_name for modeling.
        """
        getter_function = np.median

        if use_mean:
            getter_function = np.mean

        static_model = self._get_model_from_dict(self.by_name, getter_function)

        def static_model_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_model_getter

    def get_param_lut(self, fallback = False):
        """
        Get parameter-look-up-table model function: name, attribute, parameter values -> model value.

        The function can only give model values for parameter combinations
        present in by_param. By default, it raises KeyError for other values.

        arguments:
        fallback -- Fall back to the (non-parameter-aware) static model when encountering unknown parameter values
        """
        static_model = self._get_model_from_dict(self.by_name, np.median)
        lut_model = self._get_model_from_dict(self.by_param, np.median)

        def lut_median_getter(name, key, param, arg = [], **kwargs):
            param.extend(map(soft_cast_int, arg))
            try:
                return lut_model[(name, tuple(param))][key]
            except KeyError:
                if fallback:
                    return static_model[name][key]
                raise

        return lut_median_getter

    def get_fitted(self, safe_functions_enabled = False):
        """
        Get paramete-aware model function and model information function.

        Returns two functions:
        model_function(name, attribute, param=parameter values) -> model value.
        model_info(name, attribute) -> {'fit_result' : ..., 'function' : ... } or None
        """
        if 'fitted_model_getter' in self.cache and 'fitted_info_getter' in self.cache:
            return self.cache['fitted_model_getter'], self.cache['fitted_info_getter']

        static_model = self._get_model_from_dict(self.by_name, np.median)
        param_model = dict([[name, {}] for name in self.by_name.keys()])
        paramfit = ParallelParamFit(self.by_param)

        for name in self.by_name.keys():
            for attribute in self.by_name[name]['attributes']:
                for param_index, param in enumerate(self.parameters):
                    if self.stats.depends_on_param(name, attribute, param):
                        paramfit.enqueue(name, attribute, param_index, param, False)
                if arg_support_enabled and name in self._num_args:
                    for arg_index in range(self._num_args[name]):
                        if self.stats.depends_on_arg(name, attribute, arg_index):
                            paramfit.enqueue(name, attribute, len(self.parameters) + arg_index, arg_index, False)

        paramfit.fit()

        for name in self.by_name.keys():
            num_args = 0
            if name in self._num_args:
                num_args = self._num_args[name]
            for attribute in self.by_name[name]['attributes']:
                fit_result = get_fit_result(paramfit.results, name, attribute, self.verbose)

                if (name, attribute) in self.function_override:
                    function_str = self.function_override[(name, attribute)]
                    x = AnalyticFunction(function_str, self.parameters, num_args)
                    x.fit(self.by_param, name, attribute)
                    if x.fit_success:
                        param_model[name][attribute] = {
                            'fit_result': fit_result,
                            'function' : x
                        }
                elif len(fit_result.keys()):
                    x = analytic.function_powerset(fit_result, self.parameters, num_args)
                    x.fit(self.by_param, name, attribute)

                    if x.fit_success:
                        param_model[name][attribute] = {
                            'fit_result': fit_result,
                            'function' : x
                        }

        def model_getter(name, key, **kwargs):
            if 'arg' in kwargs and 'param' in kwargs:
                kwargs['param'].extend(map(soft_cast_int, kwargs['arg']))
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

    def assess(self, model_function):
        """
        Calculate MAE, SMAPE, etc. of model_function for each by_name entry.

        state/transition/... name and parameter values are fed into model_function.
        The by_name entries of this AnalyticModel are used as ground truth and
        compared with the values predicted by model_function.

        For proper model assessments, the data used to generate model_function
        and the data fed into this AnalyticModel instance must be mutually
        exclusive (e.g. by performing cross validation). Otherwise,
        overfitting cannot be detected.
        """
        detailed_results = {}
        for name, elem in sorted(self.by_name.items()):
            detailed_results[name] = {}
            for attribute in elem['attributes']:
                predicted_data = np.array(list(map(lambda i: model_function(name, attribute, param=elem['param'][i]), range(len(elem[attribute])))))
                measures = regression_measures(predicted_data, elem[attribute])
                detailed_results[name][attribute] = measures

        return {
            'by_name' : detailed_results,
        }

    def to_json(self):
        # TODO
        pass


def _add_trace_data_to_aggregate(aggregate, key, element):
    # Only cares about element['isa'], element['offline_aggregates'], and
    # element['plan']['level']
    if not key in aggregate:
        aggregate[key] = {
            'isa' : element['isa']
        }
        for datakey in element['offline_aggregates'].keys():
            aggregate[key][datakey] = []
        if element['isa'] == 'state':
            aggregate[key]['attributes'] = ['power']
        else:
            # TODO do not hardcode values
            aggregate[key]['attributes'] = ['duration', 'energy', 'rel_energy_prev', 'rel_energy_next']
            # Uncomment this line if you also want to analyze mean transition power
            #aggrgate[key]['attributes'].append('power')
            if 'plan' in element and element['plan']['level'] == 'epilogue':
                aggregate[key]['attributes'].insert(0, 'timeout')
        attributes = aggregate[key]['attributes'].copy()
        for attribute in attributes:
            if attribute not in element['offline_aggregates']:
                aggregate[key]['attributes'].remove(attribute)
    for datakey, dataval in element['offline_aggregates'].items():
        aggregate[key][datakey].extend(dataval)


def pta_trace_to_aggregate(traces, ignore_trace_indexes = []):
    u"""
    Convert preprocessed DFA traces from peripherals/drivers to by_name aggregate for PTAModel.

    arguments:
    traces -- [ ... Liste von einzelnen Läufen (d.h. eine Zustands- und Transitionsfolge UNINITIALIZED -> foo -> FOO -> bar -> BAR -> ...)
        Jeder Lauf:
        - id: int Nummer des Laufs, beginnend bei 1
        - trace: [ ... Liste von Zuständen und Transitionen
            Jeweils:
            - name: str Name
            - isa: str state // transition
            - parameter: { ... globaler Parameter: aktueller wert. null falls noch nicht eingestellt }
            - args: [ Funktionsargumente, falls isa == 'transition' ]
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
        ]
    ]
    ignore_trace_indexes -- list of trace indexes. The corresponding taces will be ignored.

    returns a tuple of three elements:
    by_name -- measurements aggregated by state/transition name, annotated with parameter values
    parameter_names -- list of parameter names
    arg_count -- dict mapping transition names to the number of arguments of their corresponding driver function

    by_name layout:
    Dictionary with one key per state/transition ('send', 'TX', ...).
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
    arg_count = dict()
    by_name = dict()
    parameter_names = sorted(traces[0]['trace'][0]['parameter'].keys())
    for run in traces:
        if run['id'] not in ignore_trace_indexes:
            for elem in run['trace']:
                if elem['isa'] == 'transition' and not elem['name'] in arg_count and 'args' in elem:
                    arg_count[elem['name']] = len(elem['args'])
                if elem['name'] != 'UNINITIALIZED':
                    _add_trace_data_to_aggregate(by_name, elem['name'], elem)
    for elem in by_name.values():
            for key in elem['attributes']:
                elem[key] = np.array(elem[key])
    return by_name, parameter_names, arg_count


class PTAModel:
    u"""
    Parameter-aware PTA-based energy model.

    Supports both static and parameter-based model attributes, and automatic detection of parameter-dependence.

    The model heavily relies on two internal data structures:
    PTAModel.by_name and PTAModel.by_param.

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

    def __init__(self, by_name, parameters, arg_count, traces = [], ignore_trace_indexes = [], discard_outliers = None, function_override = {}, verbose = True, use_corrcoef = False, pta = None):
        """
        Prepare a new PTA energy model.

        Actual model generation is done on-demand by calling the respective functions.

        arguments:
        by_name -- state/transition measurements aggregated by name, as returned by pta_trace_to_aggregate.
        parameters -- list of parameter names, as returned by pta_trace_to_aggregate
        arg_count -- function arguments, as returned by pta_trace_to_aggregate
        traces -- list of preprocessed DFA traces, as returned by RawData.get_preprocessed_data()
        ignore_trace_indexes -- list of trace indexes. The corresponding traces will be ignored.
        discard_outliers -- currently not supported: threshold for outlier detection and removel (float).
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
        pta -- hardware model as `PTA` object
        """
        self.by_name = by_name
        self.by_param = by_name_to_by_param(by_name)
        self._parameter_names = sorted(parameters)
        self._num_args = arg_count
        self._use_corrcoef = use_corrcoef
        self.traces = traces
        self.stats = ParamStats(self.by_name, self.by_param, self._parameter_names, self._num_args, self._use_corrcoef, verbose = verbose)
        self.cache = {}
        np.seterr('raise')
        self._outlier_threshold = discard_outliers
        self.function_override = function_override.copy()
        self.verbose = verbose
        self.pta = pta
        self.ignore_trace_indexes = ignore_trace_indexes
        self._aggregate_to_ndarray(self.by_name)

    def _aggregate_to_ndarray(self, aggregate):
        for elem in aggregate.values():
            for key in elem['attributes']:
                elem[key] = np.array(elem[key])

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

    def get_static(self, use_mean=False):
        """
        Get static model function: name, attribute -> model value.

        Uses the median of by_name for modeling, unless `use_mean` is set.
        """
        getter_function = np.median

        if use_mean:
            getter_function = np.mean

        static_model = self._get_model_from_dict(self.by_name, getter_function)

        def static_model_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_model_getter

    def get_param_lut(self, fallback = False):
        """
        Get parameter-look-up-table model function: name, attribute, parameter values -> model value.

        The function can only give model values for parameter combinations
        present in by_param. By default, it raises KeyError for other values.

        arguments:
        fallback -- Fall back to the (non-parameter-aware) static model when encountering unknown parameter values
        """
        static_model = self._get_model_from_dict(self.by_name, np.median)
        lut_model = self._get_model_from_dict(self.by_param, np.median)

        def lut_median_getter(name, key, param, arg = [], **kwargs):
            param.extend(map(soft_cast_int, arg))
            try:
                return lut_model[(name, tuple(param))][key]
            except KeyError:
                if fallback:
                    return static_model[name][key]
                raise

        return lut_median_getter

    def param_index(self, param_name):
        if param_name in self._parameter_names:
            return self._parameter_names.index(param_name)
        return len(self._parameter_names) + int(param_name)

    def param_name(self, param_index):
        if param_index < len(self._parameter_names):
            return self._parameter_names[param_index]
        return str(param_index)

    def get_fitted(self, safe_functions_enabled = False):
        """
        Get parameter-aware model function and model information function.

        Returns two functions:
        model_function(name, attribute, param=parameter values) -> model value.
        model_info(name, attribute) -> {'fit_result' : ..., 'function' : ... } or None
        """
        if 'fitted_model_getter' in self.cache and 'fitted_info_getter' in self.cache:
            return self.cache['fitted_model_getter'], self.cache['fitted_info_getter']

        static_model = self._get_model_from_dict(self.by_name, np.median)
        param_model = dict([[state_or_tran, {}] for state_or_tran in self.by_name.keys()])
        paramfit = ParallelParamFit(self.by_param)
        for state_or_tran in self.by_name.keys():
            for model_attribute in self.by_name[state_or_tran]['attributes']:
                fit_results = {}
                for parameter_index, parameter_name in enumerate(self._parameter_names):
                    if self.depends_on_param(state_or_tran, model_attribute, parameter_name):
                        paramfit.enqueue(state_or_tran, model_attribute, parameter_index, parameter_name, safe_functions_enabled)
                        for codependent_param_dict in self.stats.codependent_parameter_value_dicts(state_or_tran, model_attribute, parameter_name):
                            paramfit.enqueue(state_or_tran, model_attribute, parameter_index, parameter_name, safe_functions_enabled, codependent_param_dict)
                if arg_support_enabled and self.by_name[state_or_tran]['isa'] == 'transition':
                    for arg_index in range(self._num_args[state_or_tran]):
                        if self.depends_on_arg(state_or_tran, model_attribute, arg_index):
                            paramfit.enqueue(state_or_tran, model_attribute, len(self._parameter_names) + arg_index, arg_index, safe_functions_enabled)
        paramfit.fit()

        for state_or_tran in self.by_name.keys():
            num_args = 0
            if arg_support_enabled and self.by_name[state_or_tran]['isa'] == 'transition':
                num_args = self._num_args[state_or_tran]
            for model_attribute in self.by_name[state_or_tran]['attributes']:
                fit_results = get_fit_result(paramfit.results, state_or_tran, model_attribute, self.verbose)

                for parameter_name in self._parameter_names:
                    if self.depends_on_param(state_or_tran, model_attribute, parameter_name):
                        for codependent_param_dict in self.stats.codependent_parameter_value_dicts(state_or_tran, model_attribute, parameter_name):
                            pass
                            # FIXME get_fit_result hat ja gar keinen Parameter als Argument...

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
            if 'arg' in kwargs and 'param' in kwargs:
                kwargs['param'].extend(map(soft_cast_int, kwargs['arg']))
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
        self.pta.update(static_model, param_info)
        return self.pta.to_json()

    def states(self):
        """Return sorted list of state names."""
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'state', self.by_name.keys())))

    def transitions(self):
        """Return sorted list of transition names."""
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'transition', self.by_name.keys())))

    def states_and_transitions(self):
        """Return list of states and transition names."""
        ret = self.states()
        ret.extend(self.transitions())
        return ret

    def parameters(self):
        return self._parameter_names

    def attributes(self, state_or_trans):
        return self.by_name[state_or_trans]['attributes']

    def assess(self, model_function):
        """
        Calculate MAE, SMAPE, etc. of model_function for each by_name entry.

        state/transition/... name and parameter values are fed into model_function.
        The by_name entries of this PTAModel are used as ground truth and
        compared with the values predicted by model_function.

        For proper model assessments, the data used to generate model_function
        and the data fed into this AnalyticModel instance must be mutually
        exclusive (e.g. by performing cross validation). Otherwise,
        overfitting cannot be detected.
        """
        detailed_results = {}
        for name, elem in sorted(self.by_name.items()):
            detailed_results[name] = {}
            for key in elem['attributes']:
                predicted_data = np.array(list(map(lambda i: model_function(name, key, param=elem['param'][i]), range(len(elem[key])))))
                measures = regression_measures(predicted_data, elem[key])
                detailed_results[name][key] = measures

        return {
            'by_name' : detailed_results
        }

    def assess_states(self, model_function, model_attribute = 'power', distribution: dict = None):
        """
        Calculate overall model error assuming equal distribution of states
        """
        # TODO calculate mean power draw for distribution and use it to
        # calculate relative error from MAE combination
        model_quality = self.assess(model_function)
        num_states = len(self.states())
        if distribution is None:
            distribution = dict(map(lambda x: [x, 1/num_states], self.states()))

        if not np.isclose(sum(distribution.values()), 1):
            raise ValueError('distribution must be a probability distribution with sum 1')

        total_value = None
        try:
            total_value = sum(map(lambda x: model_function(x, model_attribute) * distribution[x], self.states()))
        except KeyError:
            pass

        total_error = np.sqrt(sum(map(lambda x: np.square(model_quality['by_name'][x][model_attribute]['mae'] * distribution[x]), self.states())))
        return total_error


    def assess_on_traces(self, model_function):
        """
        Calculate MAE, SMAPE, etc. of model_function for each trace known to this PTAModel instance.

        :returns: dict of `duration_by_trace`, `energy_by_trace`, `timeout_by_trace`, `rel_energy_by_trace` and `state_energy_by_trace`.
            Each entry holds regression measures for the corresponding measure. Note that the determined model quality heavily depends on the
            traces: small-ish absolute errors in states which frequently occur may have more effect than large absolute errors in rarely occuring states
        """
        model_energy_list = []
        real_energy_list = []
        model_rel_energy_list = []
        model_state_energy_list = []
        model_duration_list = []
        real_duration_list = []
        model_timeout_list = []
        real_timeout_list = []

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
                            try:
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
                            except KeyError:
                                # if states/transitions have been removed via --filter-param, this is harmless
                                pass
                    real_energy_list.append(real_energy)
                    model_energy_list.append(model_energy)
                    model_rel_energy_list.append(model_rel_energy)
                    model_state_energy_list.append(model_state_energy)
                    real_duration_list.append(real_duration)
                    model_duration_list.append(model_duration)
                    real_timeout_list.append(real_timeout)
                    model_timeout_list.append(model_timeout)

        return {
            'duration_by_trace' : regression_measures(np.array(model_duration_list), np.array(real_duration_list)),
            'energy_by_trace' : regression_measures(np.array(model_energy_list), np.array(real_energy_list)),
            'timeout_by_trace' : regression_measures(np.array(model_timeout_list), np.array(real_timeout_list)),
            'rel_energy_by_trace' : regression_measures(np.array(model_rel_energy_list), np.array(real_energy_list)),
            'state_energy_by_trace' : regression_measures(np.array(model_state_energy_list), np.array(real_energy_list)),
        }

class EnergyTraceLog:
    """
    EnergyTrace log loader for DFA traces.

    Expects an EnergyTrace log file generated via msp430-etv / energytrace-util
    and a dfatool-generated benchmark. An EnergyTrace log consits of a series
    of measurements. Each measurement has a timestamp, mean current, voltage,
    and cumulative energy since start of measurement.

    Note that the baseline power draw of board and peripherals is not subtracted
    at the moment.
    """

    def __init__(self, voltage: float, state_duration: int, transition_names: list):
        """
        Create a new EnergyTraceLog object.

        :param voltage: supply voltage [V], usually 3.3 V
        :param state_duration: state duration [ms]
        :param transition_names: list of transition names in PTA transition order.
            Needed to map barcode synchronization numbers to transitions.
        """
        self.voltage = voltage
        self.state_duration = state_duration * 1e-3
        self.transition_names = transition_names
        self.verbose = False
        self.errors = list()

        # TODO auto-detect
        self.led_power = 10e-3

        # multipass/include/object/ptalog.h#startTransition
        self.module_duration = 5e-3

        # multipass/include/object/ptalog.h#startTransition
        self.quiet_zone_duration = 60e-3

        # TODO auto-detect?
        # Note that we consider barcode duration after start, so only the
        # quiet zone -after- the code is relevant
        self.min_barcode_duration = 57 * self.module_duration + self.quiet_zone_duration
        self.max_barcode_duration = 68 * self.module_duration + self.quiet_zone_duration

    def load_data(self, log_data):
        """
        Load log data (raw energytrace .txt file, one line per event).

        :param log_data: raw energytrace log file in 4-column .txt format
        """

        if not zbar_available:
            self.errors.append('zbar module is not available. Try "apt install python3-zbar"')
            return list()

        lines = log_data.decode('ascii').split('\n')
        data_count = sum(map(lambda x: len(x) > 0 and x[0] != '#', lines))
        data_lines = filter(lambda x: len(x) > 0 and x[0] != '#', lines)

        data = np.empty((data_count, 4))

        for i, line in enumerate(data_lines):
            fields = line.split(' ')
            if len(fields) == 4:
                timestamp, current, voltage, total_energy = map(int, fields)
            elif len(fields) == 5:
                cpustate = fields[0]
                timestamp, current, voltage, total_energy = map(int, fields[1:])
            else:
                raise RuntimeError('cannot parse line "{}"'.format(line))
            data[i] = [timestamp, current, voltage, total_energy]


        self.interval_start_timestamp = data[:-1, 0] * 1e-6
        self.interval_duration = (data[1:, 0] - data[:-1, 0]) * 1e-6
        self.interval_power = ((data[1:, 3] - data[:-1, 3]) * 1e-9) / ((data[1:, 0] - data[:-1, 0]) * 1e-6)

        m_duration_us = data[-1, 0] - data[0, 0]
        m_energy_nj = data[-1, 3] - data[0, 3]

        self.sample_rate = data_count / (m_duration_us * 1e-6)

        vprint(self.verbose, 'got {} samples with {} seconds of log data ({} Hz)'.format(data_count, m_duration_us * 1e-6, self.sample_rate))

        return self.interval_start_timestamp, self.interval_duration, self.interval_power

    def ts_to_index(self, timestamp):
        """
        Convert timestamp in seconds to interval_start_timestamp / interval_duration / interval_power index.

        Returns the index of the interval which timestamp is part of.
        """
        return self._ts_to_index(timestamp, 0, len(self.interval_start_timestamp))

    def _ts_to_index(self, timestamp, left_index, right_index):
        if left_index == right_index:
            return left_index
        if left_index + 1 == right_index:
            return left_index

        mid_index = left_index + (right_index - left_index) // 2

        # I'm feeling lucky
        if timestamp > self.interval_start_timestamp[mid_index] and timestamp <= self.interval_start_timestamp[mid_index] + self.interval_duration[mid_index]:
            return mid_index

        if timestamp <= self.interval_start_timestamp[mid_index]:
            return self._ts_to_index(timestamp, left_index, mid_index)

        return self._ts_to_index(timestamp, mid_index, right_index)

    def analyze_states(self, traces, offline_index: int):
        u"""
        Split log data into states and transitions and return duration, energy, and mean power for each element.

        :param offline_index: This function uses traces[*]['trace'][*]['offline_aggregates']['duration'][offline_index] to find sync codes

        :param charges: raw charges (each element describes the charge in pJ transferred during 10 µs)
        :param trigidx: "charges" indexes corresponding to a trigger edge, see `trigger_edges`
        :param ua_func: charge(pJ) -> current(µA) function as returned by `calibration_function`

        :returns: maybe returns list of states and transitions, both starting andending with a state.
            Each element is a dict containing:
            * `isa`: 'state' or 'transition'
            * `clip_rate`: range(0..1) Anteil an Clipping im Energieverbrauch
            * `raw_mean`: Mittelwert der Rohwerte
            * `raw_std`: Standardabweichung der Rohwerte
            * `uW_mean`: Mittelwert der (kalibrierten) Leistungsaufnahme
            * `uW_std`: Standardabweichung der (kalibrierten) Leistungsaufnahme
            * `us`: Dauer
            if isa == 'transition, it also contains:
            * `timeout`: Dauer des vorherigen Zustands
            * `uW_mean_delta_prev`: Differenz zwischen uW_mean und uW_mean des vorherigen Zustands
            * `uW_mean_delta_next`: Differenz zwischen uW_mean und uW_mean des Folgezustands
        """

        first_sync = self.find_first_sync()

        energy_trace = list()

        expected_transitions = list()
        for trace_number, trace in enumerate(traces):
            for state_or_transition_number, state_or_transition in enumerate(trace['trace']):
                if state_or_transition['isa'] == 'transition':
                    try:
                        expected_transitions.append((
                            state_or_transition['name'],
                            state_or_transition['offline_aggregates']['duration'][offline_index] * 1e-6
                        ))
                    except IndexError:
                        self.errors.append('Entry #{} ("{}") in trace #{} has no duration entry for offline_index/repeat_id {}'.format(
                            state_or_transition_number, state_or_transition['name'], trace_number, offline_index))
                        return energy_trace

        next_barcode = first_sync

        for name, duration in expected_transitions:
            bc, start, stop, end = self.find_barcode(next_barcode)
            if bc is None:
                print('[!!!] did not find transition "{}"'.format(name))
                break
            next_barcode = end + self.state_duration + duration
            vprint(self.verbose, '{} barcode "{}" area: {:0.2f} .. {:0.2f} / {:0.2f} seconds'.format(offline_index, bc, start, stop, end))
            if bc != name:
                vprint(self.verbose, '[!!!] mismatch: expected "{}", got "{}"'.format(name, bc))
            vprint(self.verbose, '{} estimated transition area: {:0.3f} .. {:0.3f} seconds'.format(offline_index, end, end + duration))

            transition_start_index = self.ts_to_index(end)
            transition_done_index = self.ts_to_index(end + duration) + 1
            state_start_index = transition_done_index
            state_done_index = self.ts_to_index(end + duration + self.state_duration) + 1

            vprint(self.verbose, '{} estimated transitionindex: {:0.3f} .. {:0.3f} seconds'.format(offline_index, transition_start_index / self.sample_rate, transition_done_index / self.sample_rate))

            energy_trace.append({
                'isa': 'transition',
                'W_mean' : np.mean(self.interval_power[transition_start_index : transition_done_index]),
                'W_std' : np.std(self.interval_power[transition_start_index : transition_done_index]),
                's' : duration,
                's_coarse' : self.interval_start_timestamp[transition_done_index] - self.interval_start_timestamp[transition_start_index]

            })

            if len(energy_trace) > 1:
                energy_trace[-1]['W_mean_delta_prev'] = energy_trace[-1]['W_mean'] - energy_trace[-2]['W_mean']

            energy_trace.append({
                'isa': 'state',
                'W_mean' : np.mean(self.interval_power[state_start_index : state_done_index]),
                'W_std' : np.std(self.interval_power[state_start_index : state_done_index]),
                's' : self.state_duration,
                's_coarse' : self.interval_start_timestamp[state_done_index] - self.interval_start_timestamp[state_start_index]
            })

            energy_trace[-2]['W_mean_delta_next'] = energy_trace[-2]['W_mean'] - energy_trace[-1]['W_mean']

        expected_transition_count = len(expected_transitions)
        recovered_transition_ount = len(energy_trace) // 2

        if expected_transition_count != recovered_transition_ount:
            self.errors.append('Expected {:d} transitions, got {:d}'.format(expected_transition_count, recovered_transition_ount))

        return energy_trace

    def find_first_sync(self):
        # LED Power is approx. self.led_power W, use self.led_power/2 W above surrounding median as threshold
        sync_threshold_power = np.median(self.interval_power[: int(3 * self.sample_rate)]) + self.led_power/3
        for i, ts in enumerate(self.interval_start_timestamp):
            if ts > 2 and self.interval_power[i] > sync_threshold_power:
                return self.interval_start_timestamp[i - 300]
        return None

    def find_barcode(self, start_ts):
        """
        Return absolute position and content of the next barcode following `start_ts`.

        :param interval_ts: list of start timestamps (one per measurement interval) [s]
        :param interval_power: mean power per measurement interval [W]
        :param start_ts: timestamp at which to start looking for a barcode [s]
        """

        for i, ts in enumerate(self.interval_start_timestamp):
            if ts >= start_ts:
                start_position = i
                break

        # Lookaround: 100 ms in both directions
        lookaround = int(0.1 * self.sample_rate)


        # LED Power is approx. self.led_power W, use self.led_power/2 W above surrounding median as threshold
        sync_threshold_power = np.median(self.interval_power[start_position - lookaround : start_position + lookaround]) + self.led_power/3

        vprint(self.verbose, 'looking for barcode starting at {:0.2f} s, threshold is {:0.1f} mW'.format(start_ts, sync_threshold_power * 1e3))

        sync_area_start = None
        sync_start_ts = None
        sync_area_end = None
        sync_end_ts = None
        for i, ts in enumerate(self.interval_start_timestamp):
            if sync_area_start is None and ts >= start_ts and self.interval_power[i] > sync_threshold_power:
                sync_area_start = i - 300
                sync_start_ts = ts
            if sync_area_start is not None and sync_area_end is None and ts > sync_start_ts + self.min_barcode_duration and (ts > sync_start_ts + self.max_barcode_duration or abs(sync_threshold_power - self.interval_power[i]) > self.led_power):
                sync_area_end = i
                sync_end_ts = ts
                break

        barcode_data = self.interval_power[sync_area_start : sync_area_end]

        vprint(self.verbose, 'barcode search area: {:0.2f} .. {:0.2f} seconds ({} samples)'.format(sync_start_ts, sync_end_ts, len(barcode_data)))

        bc, start, stop, padding_bits = self.find_barcode_in_power_data(barcode_data)

        if bc is None:
            return None, None, None, None

        start_ts = self.interval_start_timestamp[sync_area_start + start]
        stop_ts = self.interval_start_timestamp[sync_area_start + stop]

        end_ts = stop_ts + self.module_duration * padding_bits + self.quiet_zone_duration

        # barcode content, barcode start timestamp, barcode stop timestamp, barcode end (stop + padding) timestamp
        return bc, start_ts, stop_ts, end_ts

    def find_barcode_in_power_data(self, barcode_data):

        min_power = np.min(barcode_data)
        max_power = np.max(barcode_data)

        # zbar seems to be confused by measurement (and thus image) noise
        # inside of barcodes. As our barcodes are only 1px high, this is
        # likely not trivial to fix.
        # -> Create a black and white (not grayscale) image to avoid this.
        # Unfortunately, this decreases resilience against background noise
        # (e.g. a not-exactly-idle peripheral device or CPU interrupts).
        image_data = np.around(1 - ((barcode_data - min_power) / (max_power - min_power)))
        image_data *= 255

        # zbar only returns the complete barcode position if it is at least
        # two pixels high. For a 1px barcode, it only returns its right border.

        width = len(image_data)
        height = 2

        image_data = bytes(map(int, image_data)) * height

        #img = Image.frombytes('L', (width, height), image_data).resize((width, 100))
        #img.save('/tmp/test-{}.png'.format(os.getpid()))

        zbimg = zbar.Image(width, height, 'Y800', image_data)
        scanner = zbar.ImageScanner()
        scanner.parse_config('enable')

        if scanner.scan(zbimg):
            sym, = zbimg.symbols
            content = sym.data
            try:
                sym_start = sym.location[1][0]
            except IndexError:
                sym_start = 0
            sym_end = sym.location[0][0]

            match = re.fullmatch(r'T(\d+)', content)
            if match:
                content = self.transition_names[int(match.group(1))]

            # PTALog barcode generation operates on bytes, so there may be
            # additional non-barcode padding (encoded as LED off / image white).
            # Calculate the amount of extra bits to determine the offset until
            # the transition starts.
            padding_bits = len(Code128(sym.data, charset='B').modules) % 8

            # sym_start leaves out the first two bars, but we don't do anything about that here
            # sym_end leaves out the last three bars, each of which is one padding bit long.
            # as a workaround, we unconditionally increment padding_bits by three.
            padding_bits += 3

            return content, sym_start, sym_end, padding_bits
        else:
            vprint(self.verbose, 'unable to find barcode')
            return None, None, None, None



class MIMOSA:
    """
    MIMOSA log loader for DFA traces with auto-calibration.

    Expects a MIMOSA log file generated via dfatool and a dfatool-generated
    benchmark. A MIMOSA log consists of a series of measurements. Each measurement
    gives the total charge (in pJ) and binary buzzer/trigger value during a 10µs interval.
    
    There must be a calibration run consisting of at least two seconds with disconnected DUT,
    two seconds with 1 kOhm (984 Ohm), and two seconds with 100 kOhm (99013 Ohm) resistor at
    the start. The first ten seconds of data are reserved for calbiration and must not contain
    measurements, as trigger/buzzer signals are ignored in this time range.
    
    Resulting data is a list of state/transition/state/transition/... measurements.
    """

    def __init__(self, voltage: float, shunt: int, verbose = True):
        """
        Initialize MIMOSA loader for a specific voltage and shunt setting.

        :param voltage: MIMOSA DUT supply voltage (V)
        :para mshunt: MIMOSA Shunt (Ohms)
        :param verbose: print notices about invalid data on STDOUT?
        """
        self.voltage = voltage
        self.shunt = shunt
        self.verbose = verbose
        self.r1 = 984 # "1k"
        self.r2 = 99013 # "100k"
        self.errors = list()

    def charge_to_current_nocal(self, charge):
        u"""
        Convert charge per 10µs (in pJ) to mean currents (in µA) without accounting for calibration.

        :param charge: numpy array of charges (pJ per 10µs) as returned by `load_data` or `load_file`

        :returns: numpy array of mean currents (µA per 10µs)
        """
        ua_max = 1.836 / self.shunt * 1000000
        ua_step = ua_max / 65535
        return charge * ua_step

    def _load_tf(self, tf):
        u"""
        Load MIMOSA log data from an open `tarfile` instance.

        :param tf: `tarfile` instance

        :returns: (numpy array of charges (pJ per 10µs), numpy array of triggers (0/1 int, per 10µs))
        """
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
        u"""
        Load MIMOSA log data from a MIMOSA log file passed as raw byte string

        :param raw_data: MIMOSA log file, passed as raw byte string

        :returns: (numpy array of charges (pJ per 10µs), numpy array of triggers (0/1 int, per 10µs))
        """
        with io.BytesIO(raw_data) as data_object:
            with tarfile.open(fileobj = data_object) as tf:
                return self._load_tf(tf)

    def load_file(self, filename):
        u"""
        Load MIMOSA log data from a MIMOSA log file

        :param filename: MIMOSA log file

        :returns: (numpy array of charges (pJ per 10µs), numpy array of triggers (0/1 int, per 10µs))
        """
        with tarfile.open(filename) as tf:
            return self._load_tf(tf)

    def currents_nocal(self, charges):
        u"""
        Convert charges (pJ per 10µs) to mean currents without accounting for calibration.
        
        :param charges: numpy array of charges (pJ per 10µs)

        :returns: numpy array of currents (mean µA per 10µs)"""
        ua_max = 1.836 / self.shunt * 1000000
        ua_step = ua_max / 65535
        return charges.astype(np.double) * ua_step

    def trigger_edges(self, triggers):
        """
        Return indexes of trigger edges (both 0->1 and 1->0) in log data.

        Ignores the first 10 seconds, which are used for calibration and may
        contain bogus triggers due to DUT resets.

        :param triggers: trigger array (int, 0/1) as returned by load_data

        :returns: list of int (trigger indices, e.g. [2000000, ...] means the first trigger appears in charges/currents interval 2000000 -> 20s after start of measurements. Keep in mind that each interval is 10µs long, not 1µs, so index values are not µs timestamps)
        """
        trigidx = []

        if len(triggers) < 1000000:
            self.errors.append('MIMOSA log is too short')
            return trigidx

        prevtrig = triggers[999999]

        # if the first trigger is high (i.e., trigger/buzzer pin is active before the benchmark starts),
        # something went wrong and are unable to determine when the first
        # transition starts.
        if prevtrig != 0:
            self.errors.append('Unable to find start of first transition (log starts with trigger == {} != 0)'.format(prevtrig))

        # if the last trigger is high (i.e., trigger/buzzer pin is active when the benchmark ends),
        # it terminated in the middle of a transition -- meaning that it was not
        # measured in its entirety.
        if triggers[-1] != 0:
            self.errors.append('Log ends during a transition'.format(prevtrig))

        # the device is reset for MIMOSA calibration in the first 10s and may
        # send bogus interrupts -> bogus triggers
        for i in range(1000000, triggers.shape[0]):
            trig = triggers[i]
            if trig != prevtrig:
                # Due to MIMOSA's integrate-read-reset cycle, the charge/current
                # interval belonging to this trigger comes two intervals (20µs) later
                trigidx.append(i+2)
            prevtrig = trig
        return trigidx

    def calibration_edges(self, currents):
        u"""
        Return start/stop indexes of calibration measurements.

        :param currents: uncalibrated currents as reported by MIMOSA. For best results,
            it may help to use a running mean, like so:
            `currents = running_mean(currents_nocal(..., 10))`

        :returns: indices of calibration events in MIMOSA data:
            (disconnect start, disconnect stop, R1 (1k) start, R1 (1k) stop, R2 (100k) start, R2 (100k) stop)
            indices refer to charges/currents arrays, so 0 refers to the first 10µs interval, 1 to the second, and so on.
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
        Calculate calibration function from previously determined calibration edges.

        :param charges: raw charges from MIMOSA
        :param cal_edges: calibration edges as returned by calibration_edges

        :returns: (calibration_function, calibration_data):
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
        cal_r1_mean = np.mean(chg_r1)
        cal_r2_mean = np.mean(chg_r2)

        ua_r1 = self.voltage / self.r1 * 1000000
        ua_r2 = self.voltage / self.r2 * 1000000

        if cal_r2_mean > cal_0_mean:
            b_lower = (ua_r2 - 0) / (cal_r2_mean - cal_0_mean)
        else:
            vprint(self.verbose, '[W] 0 uA == %.f uA during calibration' % (ua_r2))
            b_lower = 0

        b_upper = (ua_r1 - ua_r2) / (cal_r1_mean - cal_r2_mean)

        a_lower = -b_lower * cal_0_mean
        a_upper = -b_upper * cal_r2_mean

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
        Split log data into states and transitions and return duration, energy, and mean power for each element.

        :param charges: raw charges (each element describes the charge in pJ transferred during 10 µs)
        :param trigidx: "charges" indexes corresponding to a trigger edge, see `trigger_edges`
        :param ua_func: charge(pJ) -> current(µA) function as returned by `calibration_function`

        :returns: list of states and transitions, both starting andending with a state.
            Each element is a dict containing:
            * `isa`: 'state' or 'transition'
            * `clip_rate`: range(0..1) Anteil an Clipping im Energieverbrauch
            * `raw_mean`: Mittelwert der Rohwerte
            * `raw_std`: Standardabweichung der Rohwerte
            * `uW_mean`: Mittelwert der (kalibrierten) Leistungsaufnahme
            * `uW_std`: Standardabweichung der (kalibrierten) Leistungsaufnahme
            * `us`: Dauer
            if isa == 'transition, it also contains:
            * `timeout`: Dauer des vorherigen Zustands
            * `uW_mean_delta_prev`: Differenz zwischen uW_mean und uW_mean des vorherigen Zustands
            * `uW_mean_delta_next`: Differenz zwischen uW_mean und uW_mean des Folgezustands
        """
        previdx = 0
        is_state = True
        iterdata = []

        # The last state (between the last transition and end of file) may also
        # be important. Pretend it ends when the log ends.
        trigger_indices = trigidx.copy()
        trigger_indices.append(len(charges))

        for idx in trigger_indices:
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
