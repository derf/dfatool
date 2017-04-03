#!/usr/bin/env python3

import csv
from itertools import chain, combinations
import json
import numpy as np
import os
from scipy.cluster.vq import kmeans2
import struct
import sys
import tarfile

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def is_numeric(n):
    try:
        int(n)
        return True
    except ValueError:
        return False

def aggregate_measures(aggregate, actual):
    aggregate_array = np.array([aggregate] * len(actual))
    return regression_measures(aggregate_array, np.array(actual))

def regression_measures(predicted, actual):
    deviations = predicted - actual
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

    def load_data(self, filename):
        with tarfile.open(filename) as tf:
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
        return (charges, triggers)

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

        b_lower = (ua_r2 - 0) / (cal_r2_mean - cal_0_mean)
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
                data['uW_mean_delta'] = data['uW_mean'] - iterdata[-1]['uW_mean']
                data['timeout'] = iterdata[-1]['us']

            iterdata.append(data)

            previdx = idx
            is_state = not is_state
        return iterdata
