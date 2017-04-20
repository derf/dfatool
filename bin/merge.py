#!/usr/bin/env python3

import getopt
import json
import numpy as np
import os
import re
import sys
import plotter
from copy import deepcopy
from dfatool import aggregate_measures, regression_measures, is_numeric, powerset
from dfatool import append_if_set, mean_or_none
from matplotlib.patches import Polygon
from scipy import optimize

opts = {}

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w") as f:
        return json.dump(data, f)

def print_data(aggregate):
    for key in sorted(aggregate.keys()):
        data = aggregate[key]
        name, params = key
        print("%s @ %s : ~ = %.f (%.f, %.f)  µ_σ_outer = %.f  n = %d" %
            (name, params, np.median(data['means']), np.percentile(data['means'], 25),
                np.percentile(data['means'], 75), np.mean(data['stds']), len(data['means'])))

def flatten(somelist):
    return [item for sublist in somelist for item in sublist]

def mimosa_data(elem):
    means = [x['uW_mean'] for x in elem['offline']]
    durations = [x['us'] - 20 for x in elem['offline']]
    stds = [x['uW_std'] for x in elem['offline']]
    energies = [x['uW_mean'] * (x['us'] - 20) for x in elem['offline']]
    clips = [x['clip_rate'] for x in elem['offline']]
    substate_thresholds = []
    substate_data = []
    timeouts = []
    rel_energies_prev = []
    rel_energies_next = []
    if 'timeout' in elem['offline'][0]:
        timeouts = [x['timeout'] for x in elem['offline']]
    if 'uW_mean_delta_prev' in elem['offline'][0]:
        rel_energies_prev = [x['uW_mean_delta_prev'] * (x['us'] - 20) for x in elem['offline']]
    if 'uW_mean_delta_next' in elem['offline'][0]:
        rel_energies_next = [x['uW_mean_delta_next'] * (x['us'] - 20) for x in elem['offline']]
    for x in elem['offline']:
        if 'substates' in x:
            substate_thresholds.append(x['substates']['threshold'])
            substate_data.append(x['substates']['states'])

    return (means, stds, durations, energies, rel_energies_prev,
        rel_energies_next, clips, timeouts, substate_thresholds)

def online_data(elem):
    means = [int(x['power']) for x in elem['online']]
    durations = [int(x['time']) for x in elem['online']]

    return means, durations

# parameters = statistic variables such as txpower, bitrate etc.
# variables = function variables/parameters set by linear regression
def str_to_param_function(function_string, parameters, variables):
    rawfunction = function_string
    dependson = [False] * len(parameters)

    for i in range(len(parameters)):
        if rawfunction.find("global(%s)" % (parameters[i])) >= 0:
            dependson[i] = True
            rawfunction = rawfunction.replace("global(%s)" % (parameters[i]), "arg[%d]" % (i))
        if rawfunction.find("local(%s)" % (parameters[i])) >= 0:
            dependson[i] = True
            rawfunction = rawfunction.replace("local(%s)" % (parameters[i]), "arg[%d]" % (i))
    for i in range(len(variables)):
        rawfunction = rawfunction.replace("param(%d)" % (i), "param[%d]" % (i))
    fitfunc = eval("lambda param, arg: " + rawfunction);

    return fitfunc, dependson

def mk_function_data(name, paramdata, parameters, dependson, datatype):
    X = [[] for i in range(len(parameters))]
    Xm = [[] for i in range(len(parameters))]
    Y = []
    Ym = []

    num_valid = 0
    num_total = 0

    for key, val in paramdata.items():
        if key[0] == name and len(key[1]) == len(parameters):
            valid = True
            num_total += 1
            for i in range(len(parameters)):
                if dependson[i] and not is_numeric(key[1][i]):
                    valid = False
            if valid:
                num_valid += 1
                Y.extend(val[datatype])
                Ym.append(np.median(val[datatype]))
                for i in range(len(parameters)):
                    if dependson[i] or is_numeric(key[1][i]):
                        X[i].extend([float(key[1][i])] * len(val[datatype]))
                        Xm[i].append(float(key[1][i]))
                    else:
                        X[i].extend([0] * len(val[datatype]))
                        Xm[i].append(0)

    for i in range(len(parameters)):
        X[i] = np.array(X[i])
        Xm[i] = np.array(Xm[i])
    X = tuple(X)
    Xm = tuple(Xm)
    Y = np.array(Y)
    Ym = np.array(Ym)

    return X, Y, Xm, Ym, num_valid, num_total

def raw_num0_8(num):
    return (8 - num1(num))

def raw_num0_16(num):
    return (16 - num1(num))

def raw_num1(num):
    return bin(int(num)).count("1")

num0_8 = np.vectorize(raw_num0_8)
num0_16 = np.vectorize(raw_num0_16)
num1 = np.vectorize(raw_num1)

def try_fits(name, datatype, paramidx, paramdata):
    functions = {
        'linear' : lambda param, arg: param[0] + param[1] * arg,
        'logarithmic' : lambda param, arg: param[0] + param[1] * np.log(arg),
        'logarithmic1' : lambda param, arg: param[0] + param[1] * np.log(arg + 1),
        'exponential' : lambda param, arg: param[0] + param[1] * np.exp(arg),
        #'polynomial' : lambda param, arg: param[0] + param[1] * arg + param[2] * arg ** 2,
        'square' : lambda param, arg: param[0] + param[1] * arg ** 2,
        'fractional' : lambda param, arg: param[0] + param[1] / arg,
        'sqrt' : lambda param, arg: param[0] + param[1] * np.sqrt(arg),
        'num0_8' : lambda param, arg: param[0] + param[1] * num0_8(arg),
        'num0_16' : lambda param, arg: param[0] + param[1] * num0_16(arg),
        'num1' : lambda param, arg: param[0] + param[1] * num1(arg),
    }
    results = dict([[key, []] for key in functions.keys()])
    errors = {}

    allvalues = [(*key[1][:paramidx], *key[1][paramidx+1:]) for key in paramdata.keys() if key[0] == name]
    allvalues = list(set(allvalues))

    for value in allvalues:
        X = []
        Xm = []
        Y = []
        Ym = []
        num_valid = 0
        num_total = 0
        for key, val in paramdata.items():
            if key[0] == name and len(key[1]) > paramidx and (*key[1][:paramidx], *key[1][paramidx+1:]) == value:
                num_total += 1
                if is_numeric(key[1][paramidx]):
                    num_valid += 1
                    Y.extend(val[datatype])
                    Ym.append(np.median(val[datatype]))
                    X.extend([float(key[1][paramidx])] * len(val[datatype]))
                    Xm.append(float(key[1][paramidx]))
                    if float(key[1][paramidx]) == 0:
                        functions.pop('fractional', None)
                    if float(key[1][paramidx]) <= 0:
                        functions.pop('logarithmic', None)
                    if float(key[1][paramidx]) < 0:
                        functions.pop('logarithmic1', None)
                        functions.pop('sqrt', None)
                    if float(key[1][paramidx]) > 64:
                        functions.pop('exponential', None)

        # there should be -at least- two values when fitting
        if num_valid > 1:
            Y = np.array(Y)
            Ym = np.array(Ym)
            X = np.array(X)
            Xm = np.array(Xm)
            for kind, function in functions.items():
                results[kind] = {}
                errfunc = lambda P, X, y: function(P, X) - y
                try:
                    res = optimize.least_squares(errfunc, [0, 1], args=(X, Y), xtol=2e-15)
                    measures = regression_measures(function(res.x, X), Y)
                    for k, v in measures.items():
                        if not k in results[kind]:
                            results[kind][k] = []
                        results[kind][k].append(v)
                except:
                    pass

    for function_name, result in results.items():
        if len(result) > 0 and function_name in functions:
            errors[function_name] = {}
            for measure in result.keys():
                errors[function_name][measure] = np.mean(result[measure])

    return errors

def fit_function(function, name, datatype, parameters, paramdata, xaxis=None, yaxis=None):

    variables = list(map(lambda x: float(x), function['params']))
    fitfunc, dependson = str_to_param_function(function['raw'], parameters, variables)

    X, Y, Xm, Ym, num_valid, num_total = mk_function_data(name, paramdata, parameters, dependson, datatype)

    if num_valid > 0:

        if num_valid != num_total:
            num_invalid = num_total - num_valid
            print("Warning: fit(%s): %d of %d states had incomplete parameter hashes" % (name, num_invalid, len(paramdata)))

        errfunc = lambda P, X, y: fitfunc(P, X) - y
        try:
            res = optimize.least_squares(errfunc, variables, args=(X, Y), xtol=2e-15) # loss='cauchy'
        except ValueError as err:
            function['error'] = str(err)
            return
        #x1 = optimize.curve_fit(lambda param, *arg: fitfunc(param, arg), X, Y, functionparams)
        measures = regression_measures(fitfunc(res.x, X), Y)

        if res.status <= 0:
            function['error'] = res.message
            return

        if 'fit' in opts:
            for i in range(len(parameters)):
                plotter.plot_param_fit(function['raw'], name, fitfunc, res.x, parameters, datatype, i, X, Y, xaxis, yaxis)

        function['params'] = list(res.x)
        function['fit'] = measures

    else:
        function['error'] = 'log contained no numeric parameters'

def assess_function(function, name, datatype, parameters, paramdata):

    variables = list(map(lambda x: float(x), function['params']))
    fitfunc, dependson = str_to_param_function(function['raw'], parameters, variables)
    X, Y, Xm, Ym, num_valid, num_total = mk_function_data(name, paramdata, parameters, dependson, datatype)

    if num_valid > 0:
        return regression_measures(fitfunc(variables, X), Y)
    else:
        return None

def xv_assess_function(name, funbase, what, validation, mae, smape):
    goodness = assess_function(funbase, name, what, parameters, validation)
    if goodness != None:
        if not name in mae:
            mae[name] = []
        if not name in smape:
            smape[name] = []
        append_if_set(mae, goodness, 'mae')
        append_if_set(smape, goodness, 'smape')

def xv2_assess_function(name, funbase, what, validation, mae, smape, rmsd):
    goodness = assess_function(funbase, name, what, parameters, validation)
    if goodness != None:
        if goodness['mae'] < 10e9:
            mae.append(goodness['mae'])
            rmsd.append(goodness['rmsd'])
            smape.append(goodness['smape'])
        else:
            print('[!] Ignoring MAE of %d (SMAPE %.f)' % (goodness['mae'], goodness['smape']))

# Returns the values used for each parameter in the measurement, e.g.
# { 'txpower' : [1, 2, 4, 8], 'length' : [16] }
# non-numeric values such as '' are skipped
def param_values(parameters, by_param):
    paramvalues = dict([[param, set()] for param in parameters])

    for _, paramvalue in by_param.keys():
        for i, param in enumerate(parameters):
            if is_numeric(paramvalue[i]):
                paramvalues[param].add(paramvalue[i])

    return paramvalues

def param_hash(values):
    ret = {}

    for i, param in enumerate(parameters):
        ret[param] = values[i]

    return ret

# Returns the values used for each function argument in the measurement, e.g.
# { 'data': [], 'length' : [16, 31, 32] }
# non-numeric values such as '' or 'long_test_string' are skipped
def arg_values(name, by_arg):
    TODO
    argvalues = dict([[arg, set()] for arg in parameters])

    for _, paramvalue in by_param.keys():
        for i, param in enumerate(parameters):
            if is_numeric(paramvalue[i]):
                paramvalues[param].add(paramvalue[i])

    return paramvalues

def mk_param_key(elem):
    name = elem['name']
    paramtuple = ()

    if 'parameter' in elem:
        paramkeys = sorted(elem['parameter'].keys())
        paramtuple = tuple([elem['parameter'][x] for x in paramkeys])

    return (name, paramtuple)

def mk_arg_key(elem):
    name = elem['name']
    argtuple = ()

    if 'args' in elem:
        argtuple = tuple(elem['args'])

    return (name, argtuple)

def add_data_to_aggregate(aggregate, key, isa, data):
    if not key in aggregate:
        aggregate[key] = {
            'isa' : isa,
        }
        for datakey in data.keys():
            aggregate[key][datakey] = []
    for datakey, dataval in data.items():
        aggregate[key][datakey].extend(dataval)

def fake_add_data_to_aggregate(aggregate, key, isa, database, idx):
    timeout_val = []
    if len(database['timeouts']):
        timeout_val = [database['timeouts'][idx]]
    rel_energy_p_val = []
    if len(database['rel_energies_prev']):
        rel_energy_p_val = [database['rel_energies_prev'][idx]]
    rel_energy_n_val = []
    if len(database['rel_energies_next']):
        rel_energy_n_val = [database['rel_energies_next'][idx]]
    add_data_to_aggregate(aggregate, key, isa, {
        'means' : [database['means'][idx]],
        'stds' : [database['stds'][idx]],
        'durations' : [database['durations'][idx]],
        'energies' : [database['energies'][idx]],
        'rel_energies_prev' : rel_energy_p_val,
        'rel_energies_next' : rel_energy_n_val,
        'clip_rate' : [database['clip_rate'][idx]],
        'timeouts' : timeout_val,
    })

def weight_by_name(aggdata):
    total = {}
    count = {}
    weight = {}
    for key in aggdata.keys():
        if not key[0] in total:
            total[key[0]] = 0
        total[key[0]] += len(aggdata[key]['means'])
        count[key] = len(aggdata[key]['means'])
    for key in aggdata.keys():
        weight[key] = float(count[key]) / total[key[0]]
    return weight

# returns the mean standard deviation of all measurements of 'what'
# (e.g. power consumption or timeout) for state/transition 'name' where
# parameter 'index' is dynamic and all other parameters are fixed.
# I.e., if parameters are a, b, c ∈ {1,2,3} and 'index' corresponds to b', then
# this function returns the mean of the standard deviations of (a=1, b=*, c=1),
# (a=1, b=*, c=2), and so on
def mean_std_by_param(data, keys, name, what, index):
    partitions = []
    for key in keys:
        partition = []
        for k, v in data.items():
            if (*k[1][:index], *k[1][index+1:]) == key and k[0] == name:
                partition.extend(v[what])
        partitions.append(partition)
    return np.mean([np.std(partition) for partition in partitions])

# returns the mean standard deviation of all measurements of 'what'
# (e.g. energy or duration) for transition 'name' where
# the 'index'th argumetn is dynamic and all other arguments are fixed.
# I.e., if arguments are a, b, c ∈ {1,2,3} and 'index' is 1, then
# this function returns the mean of the standard deviations of (a=1, b=*, c=1),
# (a=1, b=*, c=2), and so on
def mean_std_by_arg(data, keys, name, what, index):
    return mean_std_by_param(data, keys, name, what, index)

# returns the mean standard deviation of all measurements of 'what'
# (e.g. power consumption or timeout) for state/transition 'name' where the
# trace of previous transitions is fixed except for a single transition,
# whose occurence or absence is silently ignored.
# this is done separately for each transition (-> returns a dictionary)
def mean_std_by_trace_part(data, transitions, name, what):
    ret = {}
    for transition in transitions:
        keys = set(map(lambda x : (x[0], x[1], tuple([y for y in x[2] if y != transition])), data.keys()))
        ret[transition] = {}
        partitions = []
        for key in keys:
            partition = []
            for k, v in data.items():
                key_without_transition = (k[0], k[1], tuple([y for y in k[2] if y != transition]))
                if key[0] == name and key == key_without_transition:
                    partition.extend(v[what])
            if len(partition):
                partitions.append(partition)
        ret[transition] = np.mean([np.std(partition) for partition in partitions])
    return ret


def load_run_elem(index, element, trace, by_name, by_arg, by_param, by_trace):
    means, stds, durations, energies, rel_energies_prev, rel_energies_next, clips, timeouts, sub_thresholds = mimosa_data(element)

    online_means = []
    online_durations = []
    if element['isa'] == 'state':
        online_means, online_durations = online_data(element)

    if 'voltage' in opts:
        element['parameter']['voltage'] = opts['voltage']

    arg_key   = mk_arg_key(element)
    param_key = mk_param_key(element)
    pre_trace = tuple(map(lambda x : x['name'], trace[1:index:2]))
    trace_key = (*param_key, pre_trace)
    name = element['name']

    elem_data = {
        'means' : means,
        'stds' : stds,
        'durations' : durations,
        'energies' : energies,
        'rel_energies_prev' : rel_energies_prev,
        'rel_energies_next' : rel_energies_next,
        'clip_rate' : clips,
        'timeouts' : timeouts,
        'sub_thresholds' : sub_thresholds,
        'param' : [param_key[1]] * len(means),
        'online_means' : online_means,
        'online_durations' : online_durations,
    }
    add_data_to_aggregate(by_name, name, element['isa'], elem_data)
    add_data_to_aggregate(by_arg, arg_key, element['isa'], elem_data)
    add_data_to_aggregate(by_param, param_key, element['isa'], elem_data)
    add_data_to_aggregate(by_trace, trace_key, element['isa'], elem_data)

def fmap(reftype, name, funtype):
    if funtype == 'linear':
        return "%s(%s)" % (reftype, name)
    if funtype == 'logarithmic':
        return "np.log(%s(%s))" % (reftype, name)
    if funtype == 'logarithmic1':
        return "np.log(%s(%s) + 1)" % (reftype, name)
    if funtype == 'exponential':
        return "np.exp(%s(%s))" % (reftype, name)
    if funtype == 'square':
        return "%s(%s)**2" % (reftype, name)
    if funtype == 'fractional':
        return "1 / %s(%s)" % (reftype, name)
    if funtype == 'sqrt':
        return "np.sqrt(%s(%s))" % (reftype, name)
    if funtype == 'num0_8':
        return "num0_8(%s(%s))" % (reftype, name)
    if funtype == 'num0_16':
        return "num0_16(%s(%s))" % (reftype, name)
    if funtype == 'num1':
        return "num1(%s(%s))" % (reftype, name)
    return "ERROR"

def fguess_to_function(name, datatype, aggdata, parameters, paramdata, yaxis):
    best_fit = {}
    fitguess = aggdata['fit_guess']
    params = list(filter(lambda x : x in fitguess, parameters))
    if len(params) > 0:
        for param in params:
            best_fit_val = np.inf
            for func_name, fit_val in fitguess[param].items():
                if fit_val['rmsd'] < best_fit_val:
                    best_fit_val = fit_val['rmsd']
                    best_fit[param] = func_name
        buf = '0'
        pidx = 0
        for elem in powerset(best_fit.items()):
            buf += " + param(%d)" % pidx
            pidx += 1
            for fun in elem:
                buf += " * %s" % fmap('global', *fun)
        aggdata['function']['estimate'] = {
            'raw' : buf,
            'params' : list(np.ones((pidx))),
            'base' : [best_fit[param] for param in params]
        }
        fit_function(
            aggdata['function']['estimate'], name, datatype, parameters,
            paramdata, yaxis=yaxis)

def arg_fguess_to_function(name, datatype, aggdata, arguments, argdata, yaxis):
    best_fit = {}
    fitguess = aggdata['arg_fit_guess']
    args = list(filter(lambda x : x in fitguess, arguments))
    if len(args) > 0:
        for arg in args:
            best_fit_val = np.inf
            for func_name, fit_val in fitguess[arg].items():
                if fit_val['rmsd'] < best_fit_val:
                    best_fit_val = fit_val['rmsd']
                    best_fit[arg] = func_name
        buf = '0'
        pidx = 0
        for elem in powerset(best_fit.items()):
            buf += " + param(%d)" % pidx
            pidx += 1
            for fun in elem:
                buf += " * %s" % fmap('local', *fun)
        aggdata['function']['estimate_arg'] = {
            'raw' : buf,
            'params' : list(np.ones((pidx))),
            'base' : [best_fit[arg] for arg in args]
        }
        fit_function(
            aggdata['function']['estimate_arg'], name, datatype, arguments,
            argdata, yaxis=yaxis)

def param_measures(name, paramdata, key, fun):
    mae = []
    smape = []
    rmsd = []
    for pkey, pval in paramdata.items():
        if pkey[0] == name:
            # Median ist besseres Maß für MAE / SMAPE,
            # Mean ist besseres für SSR. Da least_squares SSR optimiert
            # nutzen wir hier auch Mean.
            goodness = aggregate_measures(fun(pval[key]), pval[key])
            append_if_set(mae, goodness, 'mae')
            append_if_set(rmsd, goodness, 'rmsd')
            append_if_set(smape, goodness, 'smape')
    ret = {
        'mae' : mean_or_none(mae),
        'rmsd' : mean_or_none(rmsd),
        'smape' : mean_or_none(smape)
    }

    return ret

def arg_measures(name, argdata, key, fun):
    return param_measures(name, argdata, key, fun)

def lookup_table(name, paramdata, key, fun, keyfun):
    lut = []

    for pkey, pval in paramdata.items():
        if pkey[0] == name:
            lut.append({
                'key': keyfun(pkey[1]),
                'value': fun(pval[key]),
            })

    return lut

def keydata(name, val, argdata, paramdata, tracedata, key):
    ret = {
        'count' : len(val[key]),
        'median' : np.median(val[key]),
        'mean'   : np.mean(val[key]),
        'median_by_param' : lookup_table(name, paramdata, key, np.median, param_hash),
        'mean_goodness' : aggregate_measures(np.mean(val[key]), val[key]),
        'median_goodness' : aggregate_measures(np.median(val[key]), val[key]),
        'param_mean_goodness' : param_measures(name, paramdata, key, np.mean),
        'param_median_goodness' : param_measures(name, paramdata, key, np.median),
        'std_inner' : np.std(val[key]),
        'std_param' : np.mean([np.std(paramdata[x][key]) for x in paramdata.keys() if x[0] == name]),
        'std_trace' : np.mean([np.std(tracedata[x][key]) for x in tracedata.keys() if x[0] == name]),
        'std_by_param' : {},
        'fit_guess' : {},
        'function' : {},
    }

    if val['isa'] == 'transition':
        ret['arg_mean_goodness'] = arg_measures(name, argdata, key, np.mean)
        ret['arg_median_goodness'] = arg_measures(name, argdata, key, np.median)
        ret['median_by_arg'] = lookup_table(name, argdata, key, np.median, list)
        ret['std_arg'] = np.mean([np.std(argdata[x][key]) for x in argdata.keys() if x[0] == name])
        ret['std_by_arg'] = {}
        ret['arg_fit_guess'] = {}

    return ret

def splitidx_kfold(length, num_slices):
    pairs = []
    indexes = np.arange(length)
    for i in range(0, num_slices):
        training = np.delete(indexes, slice(i, None, num_slices))
        validation = indexes[i::num_slices]
        pairs.append((training, validation))
    return pairs

def splitidx_srs(length, num_slices):
    pairs = []
    for i in range(0, num_slices):
        shuffled = np.random.permutation(np.arange(length))
        border = int(length * float(2) / 3)
        training = shuffled[:border]
        validation = shuffled[border:]
        pairs.append((training, validation))
    return pairs

def val_run(aggdata, split_fun, count):
    mae = []
    smape = []
    rmsd = []
    pairs = split_fun(len(aggdata), count)
    for i in range(0, count):
        training = aggdata[pairs[i][0]]
        validation = aggdata[pairs[i][1]]
        median = np.median(training)
        goodness = aggregate_measures(median, validation)
        append_if_set(mae, goodness, 'mae')
        append_if_set(rmsd, goodness, 'rmsd')
        append_if_set(smape, goodness, 'smape')

    mae_mean = np.mean(mae)
    rmsd_mean = np.mean(rmsd)
    if len(smape):
        smape_mean = np.mean(smape)
    else:
        smape_mean = -1

    return mae_mean, smape_mean, rmsd_mean

# by_trace is not part of the cross-validation process
def val_run_fun(aggdata, by_trace, name, key, funtype1, funtype2, splitfun, count):
    aggdata = aggdata[name]
    isa = aggdata['isa']
    mae = []
    smape = []
    rmsd = []
    estimates = []
    pairs = splitfun(len(aggdata[key]), count)
    for i in range(0, count):
        bpa_training = {}
        bpa_validation = {}

        for idx in pairs[i][0]:
            bpa_key = (name, aggdata['param'][idx])
            fake_add_data_to_aggregate(bpa_training, bpa_key, isa, aggdata, idx)
        for idx in pairs[i][1]:
            bpa_key = (name, aggdata['param'][idx])
            fake_add_data_to_aggregate(bpa_validation, bpa_key, isa, aggdata, idx)

        fake_by_name = { name : aggdata }
        ares = analyze(fake_by_name, {}, bpa_training, by_trace, parameters)
        if name in ares[isa] and funtype2 in ares[isa][name][funtype1]['function']:
            xv2_assess_function(name, ares[isa][name][funtype1]['function'][funtype2], key, bpa_validation, mae, smape, rmsd)
            if funtype2 == 'estimate':
                if 'base' in ares[isa][name][funtype1]['function'][funtype2]:
                    estimates.append(tuple(ares[isa][name][funtype1]['function'][funtype2]['base']))
                else:
                    estimates.append(None)
    return mae, smape, rmsd, estimates

# by_trace is not part of the cross-validation process
def val_run_fun_p(aggdata, by_trace, name, key, funtype1, funtype2, splitfun, count):
    aggdata = dict([[x, aggdata[x]] for x in aggdata if x[0] == name])
    isa = aggdata[list(aggdata.keys())[0]]['isa']
    mae = []
    smape = []
    rmsd = []
    estimates = []
    pairs = splitfun(len(aggdata.keys()), count) # pairs are by_param index arrays
    keys = sorted(aggdata.keys())
    for i in range(0, count):
        bpa_training = dict([[keys[x], aggdata[keys[x]]] for x in pairs[i][0]])
        bpa_validation = dict([[keys[x], aggdata[keys[x]]] for x in pairs[i][1]])
        bna_training = {}
        for val in bpa_training.values():
            for idx in range(0, len(val[key])):
                fake_add_data_to_aggregate(bna_training, name, isa, val, idx)

        ares = analyze(bna_training, {}, bpa_training, by_trace, parameters)
        if name in ares[isa] and funtype2 in ares[isa][name][funtype1]['function']:
            xv2_assess_function(name, ares[isa][name][funtype1]['function'][funtype2], key, bpa_validation, mae, smape, rmsd)
            if funtype2 == 'estimate':
                if 'base' in ares[isa][name][funtype1]['function'][funtype2]:
                    estimates.append(tuple(ares[isa][name][funtype1]['function'][funtype2]['base']))
                else:
                    estimates.append(None)
    return mae, smape, rmsd, estimates

def crossvalidate(by_name, by_param, by_trace, model, parameters):
    param_mc_count = 200
    paramv = param_values(parameters, by_param)
    for name in sorted(by_name.keys()):
        isa = by_name[name]['isa']
        by_name[name]['means'] = np.array(by_name[name]['means'])
        by_name[name]['energies'] = np.array(by_name[name]['energies'])
        by_name[name]['rel_energies_prev'] = np.array(by_name[name]['rel_energies_prev'])
        by_name[name]['rel_energies_next'] = np.array(by_name[name]['rel_energies_next'])
        by_name[name]['durations'] = np.array(by_name[name]['durations'])

        if isa == 'state':
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['means'], splitidx_srs, 200)
            print('%16s,   static        power,             Monte Carlo: MAE %8.f µW,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['means'], splitidx_kfold, 10)
            print('%16s,   static        power,             10-fold sys: MAE %8.f µW,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
        else:
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['energies'], splitidx_srs, 200)
            print('%16s,   static       energy,             Monte Carlo: MAE %8.f pJ,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['energies'], splitidx_kfold, 10)
            print('%16s,   static       energy,             10-fold sys: MAE %8.f pJ,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['rel_energies_prev'], splitidx_srs, 200)
            print('%16s,   static rel_energy_p,             Monte Carlo: MAE %8.f pJ,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['rel_energies_prev'], splitidx_kfold, 10)
            print('%16s,   static rel_energy_p,             10-fold sys: MAE %8.f pJ,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['rel_energies_next'], splitidx_srs, 200)
            print('%16s,   static rel_energy_n,             Monte Carlo: MAE %8.f pJ,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['rel_energies_next'], splitidx_kfold, 10)
            print('%16s,   static rel_energy_n,             10-fold sys: MAE %8.f pJ,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['durations'], splitidx_srs, 200)
            print('%16s,   static     duration,             Monte Carlo: MAE %8.f µs,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))
            mae_mean, smape_mean, rms_mean = val_run(by_name[name]['durations'], splitidx_kfold, 10)
            print('%16s,   static     duration,             10-fold sys: MAE %8.f µs,  SMAPE %6.2f%%,  RMS %d' % (name, mae_mean, smape_mean, rms_mean))

        def print_estimates(estimates, total):
            histogram = {}
            buf = '    '
            for estimate in estimates:
                if not estimate in histogram:
                    histogram[estimate] = 1
                else:
                    histogram[estimate] += 1
            for estimate, count in sorted(histogram.items(), key=lambda kv: kv[1], reverse=True):
                buf += '  %.f%% %s' % (count * 100 / total, estimate)
            if len(estimates):
                print(buf)

        def val_run_funs(by_name, by_trace, name, key1, key2, key3, unit):
            mae, smape, rmsd, estimates = val_run_fun(by_name, by_trace, name, key1, key2, key3, splitidx_srs, param_mc_count)
            print('%16s, %8s %12s,             Monte Carlo: MAE %8.f %s,  SMAPE %6.2f%%,  RMS %d' % (
                name, key3, key2, np.mean(mae), unit, np.mean(smape), np.mean(rmsd)))
            print_estimates(estimates, param_mc_count)
            mae, smape, rmsd, estimates = val_run_fun(by_name, by_trace, name, key1, key2, key3, splitidx_kfold, 10)
            print('%16s, %8s %12s,             10-fold sys: MAE %8.f %s,  SMAPE %6.2f%%,  RMS %d' % (
                name, key3, key2, np.mean(mae), unit, np.mean(smape), np.mean(rmsd)))
            print_estimates(estimates, 10)
            mae, smape, rmsd, estimates = val_run_fun_p(by_param, by_trace, name, key1, key2, key3, splitidx_srs, param_mc_count)
            print('%16s, %8s %12s, param-aware Monte Carlo: MAE %8.f %s,  SMAPE %6.2f%%,  RMS %d' % (
                name, key3, key2, np.mean(mae), unit, np.mean(smape), np.mean(rmsd)))
            print_estimates(estimates, param_mc_count)
            mae, smape, rmsd, estimates = val_run_fun_p(by_param, by_trace, name, key1, key2, key3, splitidx_kfold, 10)
            print('%16s, %8s %12s, param-aware 10-fold sys: MAE %8.f %s,  SMAPE %6.2f%%,  RMS %d' % (
                name, key3, key2, np.mean(mae), unit, np.mean(smape), np.mean(rmsd)))
            print_estimates(estimates, 10)

        if 'power' in model[isa][name] and 'function' in model[isa][name]['power']:
            if 'user' in model[isa][name]['power']['function']:
                val_run_funs(by_name, by_trace, name, 'means', 'power', 'user', 'µW')
            if 'estimate' in model[isa][name]['power']['function']:
                val_run_funs(by_name, by_trace, name, 'means', 'power', 'estimate', 'µW')
        if 'timeout' in model[isa][name] and 'function' in model[isa][name]['timeout']:
            if 'user' in model[isa][name]['timeout']['function']:
                val_run_funs(by_name, by_trace, name, 'timeouts', 'timeout', 'user', 'µs')
            if 'estimate' in model[isa][name]['timeout']['function']:
                val_run_funs(by_name, by_trace, name, 'timeouts', 'timeout', 'estimate', 'µs')
        if 'duration' in model[isa][name] and 'function' in model[isa][name]['duration']:
            if 'user' in model[isa][name]['duration']['function']:
                val_run_funs(by_name, by_trace, name, 'durations', 'duration', 'user', 'µs')
            if 'estimate' in model[isa][name]['duration']['function']:
                val_run_funs(by_name, by_trace, name, 'durations', 'duration', 'estimate', 'µs')
        if 'energy' in model[isa][name] and 'function' in model[isa][name]['energy']:
            if 'user' in model[isa][name]['energy']['function']:
                val_run_funs(by_name, by_trace, name, 'energies', 'energy', 'user', 'pJ')
            if 'estimate' in model[isa][name]['energy']['function']:
                val_run_funs(by_name, by_trace, name, 'energies', 'energy', 'estimate', 'pJ')
        if 'rel_energy_prev' in model[isa][name] and 'function' in model[isa][name]['rel_energy_prev']:
            if 'user' in model[isa][name]['rel_energy_prev']['function']:
                val_run_funs(by_name, by_trace, name, 'rel_energies_prev', 'rel_energy_prev', 'user', 'pJ')
            if 'estimate' in model[isa][name]['rel_energy_prev']['function']:
                val_run_funs(by_name, by_trace, name, 'rel_energies_prev', 'rel_energy_prev', 'estimate', 'pJ')
        if 'rel_energy_next' in model[isa][name] and 'function' in model[isa][name]['rel_energy_next']:
            if 'user' in model[isa][name]['rel_energy_next']['function']:
                val_run_funs(by_name, by_trace, name, 'rel_energies_next', 'rel_energy_next', 'user', 'pJ')
            if 'estimate' in model[isa][name]['rel_energy_next']['function']:
                val_run_funs(by_name, by_trace, name, 'rel_energies_next', 'rel_energy_next', 'estimate', 'pJ')

    return
    for i, param in enumerate(parameters):
        user_mae = {}
        user_smape = {}
        estimate_mae = {}
        estimate_smape = {}
        for val in paramv[param]:
            bpa_training = dict([[x, by_param[x]] for x in by_param if x[1][i] != val])
            bpa_validation = dict([[x, by_param[x]] for x in by_param if x[1][i] == val])
            to_pop = []
            for name in by_name.keys():
                if not any(map(lambda x : x[0] == name, bpa_training.keys())):
                    to_pop.append(name)
            for name in to_pop:
                by_name.pop(name, None)
            ares = analyze(by_name, {}, bpa_training, by_trace, parameters)
            for name in sorted(ares['state'].keys()):
                state = ares['state'][name]
                if 'function' in state['power']:
                    if 'user' in state['power']['function']:
                        xv_assess_function(name, state['power']['function']['user'], 'means', bpa_validation, user_mae, user_smape)
                    if 'estimate' in state['power']['function']:
                        xv_assess_function(name, state['power']['function']['estimate'], 'means', bpa_validation, estimate_mae, estimate_smape)
            for name in sorted(ares['transition'].keys()):
                trans = ares['transition'][name]
                if 'timeout' in trans and 'function' in trans['timeout']:
                    if 'user' in trans['timeout']['function']:
                        xv_assess_function(name, trans['timeout']['function']['user'], 'timeouts', bpa_validation, user_mae, user_smape)
                    if 'estimate' in trans['timeout']['function']:
                        xv_assess_function(name, trans['timeout']['function']['estimate'], 'timeouts', bpa_validation, estimate_mae, estimate_smape)

        for name in sorted(user_mae.keys()):
            if by_name[name]['isa'] == 'state':
                print('user function %s power by %s: MAE %.f µW,  SMAPE %.2f%%' % (
                    name, param, np.mean(user_mae[name]), np.mean(user_smape[name])))
            else:
                print('user function %s timeout by %s: MAE %.f µs,  SMAPE %.2f%%' % (
                    name, param, np.mean(user_mae[name]), np.mean(user_smape[name])))
        for name in sorted(estimate_mae.keys()):
            if by_name[name]['isa'] == 'state':
                print('estimate function %s power by %s: MAE %.f µW,  SMAPE %.2f%%' % (
                    name, param, np.mean(estimate_mae[name]), np.mean(estimate_smape[name])))
            else:
                print('estimate function %s timeout by %s: MAE %.f µs,  SMAPE %.2f%%' % (
                    name, param, np.mean(estimate_mae[name]), np.mean(estimate_smape[name])))

def analyze_by_param(aggval, by_param, allvalues, name, key1, key2, param, param_idx):
    aggval[key1]['std_by_param'][param] = mean_std_by_param(
        by_param, allvalues, name, key2, param_idx)
    if aggval[key1]['std_by_param'][param] > 0 and aggval[key1]['std_param'] / aggval[key1]['std_by_param'][param] < 0.6:
        aggval[key1]['fit_guess'][param] = try_fits(name, key2, param_idx, by_param)

def analyze_by_arg(aggval, by_arg, allvalues, name, key1, key2, arg_name, arg_idx):
    aggval[key1]['std_by_arg'][arg_name] = mean_std_by_arg(
        by_arg, allvalues, name, key2, arg_idx)
    if aggval[key1]['std_by_arg'][arg_name] > 0 and aggval[key1]['std_arg'] / aggval[key1]['std_by_arg'][arg_name] < 0.6:
        aggval[key1]['arg_fit_guess'][arg_name] = try_fits(name, key2, arg_idx, by_arg)

def maybe_fit_function(aggval, model, by_param, parameters, name, key1, key2, unit):
    if 'function' in model[key1] and 'user' in model[key1]['function']:
        aggval[key1]['function']['user'] = {
            'raw' : model[key1]['function']['user']['raw'],
            'params' : model[key1]['function']['user']['params'],
        }
        fit_function(
            aggval[key1]['function']['user'], name, key2, parameters, by_param,
            yaxis='%s %s by param [%s]' % (name, key1, unit))

def analyze(by_name, by_arg, by_param, by_trace, parameters):
    aggdata = {
        'state' : {},
        'transition' : {},
        'min_voltage' : min_voltage,
        'max_voltage' : max_voltage,
    }
    transition_names = list(map(lambda x: x[0], filter(lambda x: x[1]['isa'] == 'transition', by_name.items())))
    for name, val in by_name.items():
        isa = val['isa']
        model = data['model'][isa][name]

        aggdata[isa][name] = {
            'power' : keydata(name, val, by_arg, by_param, by_trace, 'means'),
            'duration' : keydata(name, val, by_arg, by_param, by_trace, 'durations'),
            'energy' : keydata(name, val, by_arg, by_param, by_trace, 'energies'),
            'clip' : {
                'mean' : np.mean(val['clip_rate']),
                'max'  : max(val['clip_rate']),
            },
            'timeout' : {},
        }

        aggval = aggdata[isa][name]
        aggval['power']['std_outer'] = np.mean(val['stds'])

        if isa == 'transition':
            aggval['rel_energy_prev'] = keydata(name, val, by_arg, by_param, by_trace, 'rel_energies_prev')
            aggval['rel_energy_next'] = keydata(name, val, by_arg, by_param, by_trace, 'rel_energies_next')
            aggval['timeout'] = keydata(name, val, by_arg, by_param, by_trace, 'timeouts')

        for i, param in enumerate(parameters):
            values = list(set([key[1][i] for key in by_param.keys() if key[0] == name and key[1][i] != '']))
            allvalues = [(*key[1][:i], *key[1][i+1:]) for key in by_param.keys() if key[0] == name]
            #allvalues = list(set(allvalues))
            if len(values) > 1:
                if isa == 'state':
                    analyze_by_param(aggval, by_param, allvalues, name, 'power', 'means', param, i)
                else:
                    analyze_by_param(aggval, by_param, allvalues, name, 'duration', 'durations', param, i)
                    analyze_by_param(aggval, by_param, allvalues, name, 'energy', 'energies', param, i)
                    analyze_by_param(aggval, by_param, allvalues, name, 'rel_energy_prev', 'rel_energies_prev', param, i)
                    analyze_by_param(aggval, by_param, allvalues, name, 'rel_energy_next', 'rel_energies_next', param, i)
                    analyze_by_param(aggval, by_param, allvalues, name, 'timeout', 'timeouts', param, i)

        if isa == 'state':
            fguess_to_function(name, 'means', aggval['power'], parameters, by_param,
                'estimated %s power by param [µW]' % name)
            maybe_fit_function(aggval, model, by_param, parameters, name, 'power', 'means', 'µW')
            if aggval['power']['std_param'] > 0 and aggval['power']['std_trace'] / aggval['power']['std_param'] < 0.5:
                aggval['power']['std_by_trace'] = mean_std_by_trace_part(by_trace, transition_names, name, 'means')
        else:
            fguess_to_function(name, 'durations', aggval['duration'], parameters, by_param,
                'estimated %s duration by param [µs]' % name)
            fguess_to_function(name, 'energies', aggval['energy'], parameters, by_param,
                'estimated %s energy by param [pJ]' % name)
            fguess_to_function(name, 'rel_energies_prev', aggval['rel_energy_prev'], parameters, by_param,
                'estimated relative_prev %s energy by param [pJ]' % name)
            fguess_to_function(name, 'rel_energies_next', aggval['rel_energy_next'], parameters, by_param,
                'estimated relative_next %s energy by param [pJ]' % name)
            fguess_to_function(name, 'timeouts', aggval['timeout'], parameters, by_param,
                'estimated %s timeout by param [µs]' % name)
            maybe_fit_function(aggval, model, by_param, parameters, name, 'duration', 'durations', 'µs')
            maybe_fit_function(aggval, model, by_param, parameters, name, 'energy', 'energies', 'pJ')
            maybe_fit_function(aggval, model, by_param, parameters, name, 'rel_energy_prev', 'rel_energies_prev', 'pJ')
            maybe_fit_function(aggval, model, by_param, parameters, name, 'rel_energy_next', 'rel_energies_next', 'pJ')
            maybe_fit_function(aggval, model, by_param, parameters, name, 'timeout', 'timeouts', 'µs')

            for i, arg in enumerate(model['parameters']):
                values = list(set([key[1][i] for key in by_arg.keys() if key[0] == name and is_numeric(key[1][i])]))
                allvalues = [(*key[1][:i], *key[1][i+1:]) for key in by_arg.keys() if key[0] == name]
                analyze_by_arg(aggval, by_arg, allvalues, name, 'duration', 'durations', arg['name'], i)
                analyze_by_arg(aggval, by_arg, allvalues, name, 'energy', 'energies', arg['name'], i)
                analyze_by_arg(aggval, by_arg, allvalues, name, 'rel_energy_prev', 'rel_energies_prev', arg['name'], i)
                analyze_by_arg(aggval, by_arg, allvalues, name, 'rel_energy_next', 'rel_energies_next', arg['name'], i)
                analyze_by_arg(aggval, by_arg, allvalues, name, 'timeout', 'timeouts', arg['name'], i)

            arguments = list(map(lambda x: x['name'], model['parameters']))
            arg_fguess_to_function(name, 'durations', aggval['duration'], arguments, by_arg,
                'estimated %s duration by arg [µs]' % name)
            arg_fguess_to_function(name, 'energies', aggval['energy'], arguments, by_arg,
                'estimated %s energy by arg [pJ]' % name)
            arg_fguess_to_function(name, 'rel_energies_prev', aggval['rel_energy_prev'], arguments, by_arg,
                'estimated relative_prev %s energy by arg [pJ]' % name)
            arg_fguess_to_function(name, 'rel_energies_next', aggval['rel_energy_next'], arguments, by_arg,
                'estimated relative_next %s energy by arg [pJ]' % name)
            arg_fguess_to_function(name, 'timeouts', aggval['timeout'], arguments, by_arg,
                'estimated %s timeout by arg [µs]' % name)

    return aggdata

try:
    raw_opts, args = getopt.getopt(sys.argv[1:], "", [
        "fit", "states", "transitions", "params", "clipping", "timing",
        "histogram", "substates", "validate", "crossvalidate", "ignore-trace-idx=", "voltage"])
    for option, parameter in raw_opts:
        optname = re.sub(r'^--', '', option)
        opts[optname] = parameter
    if 'ignore-trace-idx' in opts:
        opts['ignore-trace-idx'] = int(opts['ignore-trace-idx'])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

data = load_json(args[0])
by_name = {}
by_arg = {}
by_param = {}
by_trace = {}

if 'voltage' in opts:
    data['model']['parameter']['voltage'] = {
        'default' : float(data['setup']['mimosa_voltage']),
        'function' : None,
        'arg_name' : None,
    }

min_voltage = float(data['setup']['mimosa_voltage'])
max_voltage = float(data['setup']['mimosa_voltage'])

parameters = sorted(data['model']['parameter'].keys())

for arg in args:
    mdata = load_json(arg)
    this_voltage = float(mdata['setup']['mimosa_voltage'])
    if this_voltage > max_voltage:
        max_voltage = this_voltage
    if this_voltage < min_voltage:
        min_voltage = this_voltage
    if 'voltage' in opts:
        opts['voltage'] = this_voltage
    for runidx, run in enumerate(mdata['traces']):
        if 'ignore-trace-idx' not in opts or opts['ignore-trace-idx'] != runidx:
            for i, elem in enumerate(run['trace']):
                if elem['name'] != 'UNINITIALIZED':
                    load_run_elem(i, elem, run['trace'], by_name, by_arg, by_param, by_trace)

if 'states' in opts:
    if 'params' in opts:
        plotter.plot_states_param(data['model'], by_param)
    else:
        plotter.plot_states(data['model'], by_name)
    if 'timing' in opts:
        plotter.plot_states_duration(data['model'], by_name)
        plotter.plot_states_duration(data['model'], by_param)
    if 'clipping' in opts:
        plotter.plot_states_clips(data['model'], by_name)
if 'transitions' in opts:
    plotter.plot_transitions(data['model'], by_name)
    if 'timing' in opts:
        plotter.plot_transitions_duration(data['model'], by_name)
        plotter.plot_transitions_timeout(data['model'], by_param)
    if 'clipping' in opts:
        plotter.plot_transitions_clips(data['model'], by_name)
if 'histogram' in opts:
    for key in sorted(by_name.keys()):
        plotter.plot_histogram(by_name[key]['means'])
if 'substates' in opts:
    if 'params' in opts:
        plotter.plot_substate_thresholds_p(data['model'], by_param)
    else:
        plotter.plot_substate_thresholds(data['model'], by_name)

if 'crossvalidate' in opts:
    crossvalidate(by_name, by_param, by_trace, data['model'], parameters)
else:
    data['aggregate'] = analyze(by_name, by_arg, by_param, by_trace, parameters)

# TODO optionally also plot data points for states/transitions which do not have
# a function, but may depend on a parameter (visualization is always good!)

save_json(data, args[0])
