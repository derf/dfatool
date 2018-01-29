#!/usr/bin/env python3

import csv
from itertools import chain, combinations
import io
import json
import numpy as np
import os
from scipy.cluster.vq import kmeans2
import struct
import sys
import tarfile
from multiprocessing import Pool

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

def float_or_nan(n):
    if n == None:
        return np.nan
    try:
        return float(n)
    except ValueError:
        return np.nan

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

            if online_trace_part['isa'] != offline_trace_part['isa']:
                processed_data['error'] = 'Offline #{off_idx:d} (online {on_name:s} @ {on_idx:d}/{on_sub:d}) claims to be {off_isa:s}, but should be {on_isa:s}'.format(
                        off_idx = offline_idx, on_idx = online_run_idx,
                        on_sub = online_trace_part_idx,
                        on_name = online_trace_part['name'],
                        off_isa = offline_trace_part['isa'],
                        on_isa = online_trace_part['isa'])
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

            if not 'offline_aggregates' in online_trace_part:
                online_trace_part['offline_aggregates'] = {
                    'power' : [],
                    'duration' : [],
                    'power_std' : [],
                    'energy' : [],
                    'clipping' : [],
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
            online_trace_part['offline_aggregates']['clipping'].append(
                offline_trace_part['clip_rate'])
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

class EnergyModel:

    def __init__(self, preprocessed_data):
        self.traces = preprocessed_data
        self.by_name = {}
        self.by_arg = {}
        self.by_param = {}
        self.by_trace = {}
        np.seterr('raise')
        for runidx, run in enumerate(self.traces):
            # if opts['ignore-trace-idx'] != runidx
            for i, elem in enumerate(run['trace']):
                if elem['name'] != 'UNINITIALIZED':
                    self._load_run_elem(i, elem)
        self._aggregate_to_ndarray(self.by_name)

    def _aggregate_to_ndarray(self, aggregate):
        for elem in aggregate.values():
            for key in ['power', 'energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']:
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

    def _load_run_elem(self, i, elem):
        self._add_data_to_aggregate(self.by_name, elem['name'], elem)

    def get_static(self):
        static_model = {}
        for name, elem in self.by_name.items():
            static_model[name] = {}
            for key in ['power', 'energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']:
                if key in elem:
                    try:
                        static_model[name][key] = np.median(elem[key])
                    except RuntimeWarning:
                        print('[W] Got no data for {} {}'.format(name, key))
                    except FloatingPointError as fpe:
                        print('[W] Got no data for {} {}: {}'.format(name, key, fpe))

        def static_median_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_median_getter

    def get_static_using_mean(self):
        static_model = {}
        for name, elem in self.by_name.items():
            static_model[name] = {}
            for key in ['power', 'energy', 'duration', 'timeout', 'rel_energy_prev', 'rel_energy_next']:
                if key in elem:
                    try:
                        static_model[name][key] = np.mean(elem[key])
                    except RuntimeWarning:
                        print('[W] Got no data for {} {}'.format(name, key))
                    except FloatingPointError as fpe:
                        print('[W] Got no data for {} {}: {}'.format(name, key, fpe))

        def static_mean_getter(name, key, **kwargs):
            return static_model[name][key]

        return static_mean_getter

    def states(self):
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'state', self.by_name.keys())))

    def transitions(self):
        return sorted(list(filter(lambda k: self.by_name[k]['isa'] == 'transition', self.by_name.keys())))

    def assess(self, model_function):
        for name, elem in sorted(self.by_name.items()):
            print('{}:'.format(name))
            if elem['isa'] == 'state':
                predicted_data = np.array(list(map(lambda x: model_function(name, 'power'), elem['power'])))
                measures = regression_measures(predicted_data, elem['power'])
                if 'smape' in measures:
                    print('  power: {:.2f}% / {:.0f} µW'.format(
                        measures['smape'], measures['mae']
                    ))
                else:
                    print('  power: {:.0f} µW'.format(
                        measures['mae']
                    ))
            else:
                for key in ['duration', 'energy', 'rel_energy_prev', 'rel_energy_next']:
                    predicted_data = np.array(list(map(lambda x: model_function(name, key), elem[key])))
                    measures = regression_measures(predicted_data, elem[key])
                    if 'smape' in measures:
                        print('  {:10s}: {:.2f}% / {:.0f}'.format(
                            key, measures['smape'], measures['mae']
                        ))
                    else:
                        print('  {:10s}: {:.0f}'.format(
                            key, measures['mae']
                        ))



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
