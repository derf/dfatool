#!/usr/bin/env python3

import itertools
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Polygon

def float_or_nan(n):
    if n == None:
        return np.nan
    try:
        return float(n)
    except ValueError:
        return np.nan

def flatten(somelist):
    return [item for sublist in somelist for item in sublist]

def is_state(aggregate, name):
    return aggregate[name]['isa'] == 'state' and name != 'UNINITIALIZED'

def plot_states(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if is_state(aggregate, key)]
    data = [aggregate[key]['means'] for key in keys]
    mdata = [int(model['state'][key]['power']['static']) for key in keys]
    boxplot(keys, mdata, None, data, 'Zustand', 'µW')

def plot_transitions(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if aggregate[key]['isa'] == 'transition']
    data = [aggregate[key]['rel_energies'] for key in keys]
    mdata = [int(model['transition'][key]['rel_energy']['static']) for key in keys]
    boxplot(keys, mdata, None, data, 'Transition', 'pJ (rel)')
    data = [aggregate[key]['energies'] for key in keys]
    mdata = [int(model['transition'][key]['energy']['static']) for key in keys]
    boxplot(keys, mdata, None, data, 'Transition', 'pJ')

def plot_states_duration(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if is_state(aggregate, key)]
    data = [aggregate[key]['durations'] for key in keys]
    boxplot(keys, None, None, data, 'Zustand', 'µs')

def plot_transitions_duration(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if aggregate[key]['isa'] == 'transition']
    data = [aggregate[key]['durations'] for key in keys]
    boxplot(keys, None, None, data, 'Transition', 'µs')

def plot_transitions_timeout(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if aggregate[key]['isa'] == 'transition']
    data = [aggregate[key]['timeouts'] for key in keys]
    boxplot(keys, None, None, data, 'Timeout', 'µs')

def plot_states_clips(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if is_state(aggregate, key)]
    data = [np.array([100]) * aggregate[key]['clip_rate'] for key in keys]
    boxplot(keys, None, None, data, 'Zustand', '% Clipping')

def plot_transitions_clips(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if aggregate[key]['isa'] == 'transition']
    data = [np.array([100]) * aggregate[key]['clip_rate'] for key in keys]
    boxplot(keys, None, None, data, 'Transition', '% Clipping')

def plot_substate_thresholds(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if is_state(aggregate, key)]
    data = [aggregate[key]['sub_thresholds'] for key in keys]
    boxplot(keys, None, None, data, 'Zustand', 'substate threshold (mW/dmW)')

def plot_histogram(data):
    n, bins, patches = plt.hist(data, 1000, normed=1, facecolor='green', alpha=0.75)
    plt.show()

def plot_states_param(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if aggregate[key]['isa'] == 'state' and key[0] != 'UNINITIALIZED']
    data = [aggregate[key]['means'] for key in keys]
    mdata = [int(model['state'][key[0]]['power']['static']) for key in keys]
    boxplot(keys, mdata, None, data, 'Transition', 'µW')

def plot_attribute(aggregate, attribute, attribute_unit = '', key_filter = lambda x: True):
    """
    Boxplot measurements of a single attribute according to the partitioning provided by aggregate.

    Plots aggregate[*][attribute] with one column per aggregate key.

    arguments:
    aggregate -- measurements. aggregate[*][attribute] must be list of numbers
    attribute -- attribute to plot, e.g. 'power' or 'duration'
    attribute_init -- attribute unit for display in X axis legend
    key_filter -- if set: Only plot keys where key_filter(key) returns True
    """
    keys = list(filter(key_filter, sorted(aggregate.keys())))
    data = [aggregate[key][attribute] for key in keys]
    boxplot(keys, None, None, data, attribute, attribute_unit)

def plot_substate_thresholds_p(model, aggregate):
    keys = [key for key in sorted(aggregate.keys()) if aggregate[key]['isa'] == 'state' and key[0] != 'UNINITIALIZED']
    data = [aggregate[key]['sub_thresholds'] for key in keys]
    boxplot(keys, None, None, data, 'Zustand', '% Clipping')

def plot_y(Y, **kwargs):
    plot_xy(np.arange(len(Y)), Y, **kwargs)

def plot_xy(X, Y, xlabel = None, ylabel = None, title = None, output = None):
    fig, ax1 = plt.subplots(figsize=(10,6))
    if title != None:
        fig.canvas.set_window_title(title)
    if xlabel != None:
        ax1.set_xlabel(xlabel)
    if ylabel != None:
        ax1.set_ylabel(ylabel)
    plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.99, top = 0.99)
    plt.plot(X, Y, "bo", markersize=2)
    if output:
        plt.savefig(output)
        with open('{}.txt'.format(output), 'w') as f:
            print('X Y', file=f)
            for i in range(len(X)):
                print('{} {}'.format(X[i], Y[i]), file=f)
    else:
        plt.show()

def _param_slice_eq(a, b, index):
    return (*a[1][:index], *a[1][index+1:]) == (*b[1][:index], *b[1][index+1:]) and a[0] == b[0]

def plot_param(model, state_or_trans, attribute, param_idx, xlabel = None, ylabel = None, title = None, extra_function = None, output = None):
    fig, ax1 = plt.subplots(figsize=(10,6))
    if title != None:
        fig.canvas.set_window_title(title)
    if xlabel != None:
        ax1.set_xlabel(xlabel)
    if ylabel != None:
        ax1.set_ylabel(ylabel)
    plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.99, top = 0.99)

    param_name = model.param_name(param_idx)

    function_filename = 'plot_param_{}_{}_{}.txt'.format(state_or_trans, attribute, param_name)
    data_filename_base = 'measurements_{}_{}_{}'.format(state_or_trans, attribute, param_name)

    param_model, param_info = model.get_fitted()

    by_other_param = {}

    XX = []

    legend_sanitizer = re.compile(r'[^0-9a-zA-Z]+')

    for k, v in model.by_param.items():
        if k[0] == state_or_trans:
            other_param_key = (*k[1][:param_idx], *k[1][param_idx+1:])
            if not other_param_key in by_other_param:
                by_other_param[other_param_key] = {'X': [], 'Y': []}
            by_other_param[other_param_key]['X'].extend([float(k[1][param_idx])] * len(v[attribute]))
            by_other_param[other_param_key]['Y'].extend(v[attribute])
            XX.extend(by_other_param[other_param_key]['X'])

    XX = np.array(XX)
    x_range = int((XX.max() - XX.min()) * 10)
    xsp = np.linspace(XX.min(), XX.max(), x_range)
    YY = [xsp]
    YY_legend = [param_name]
    YY2 = []
    YY2_legend = []

    cm = plt.get_cmap('brg', len(by_other_param))
    for i, k in sorted(enumerate(by_other_param), key = lambda x: x[1]):
        v = by_other_param[k]
        v['X'] = np.array(v['X'])
        v['Y'] = np.array(v['Y'])
        plt.plot(v['X'], v['Y'], "ro", color=cm(i), markersize=3)
        YY2_legend.append(legend_sanitizer.sub('_', 'X_{}'.format(k)))
        YY2.append(v['X'])
        YY2_legend.append(legend_sanitizer.sub('_', 'Y_{}'.format(k)))
        YY2.append(v['Y'])

        sanitized_k = legend_sanitizer.sub('_', str(k))
        with open('{}_{}.txt'.format(data_filename_base, sanitized_k), 'w') as f:
            print('X Y', file=f)
            for i in range(len(v['X'])):
                print('{} {}'.format(v['X'][i], v['Y'][i]), file=f)

        #x_range = int((v['X'].max() - v['X'].min()) * 10)
        #xsp = np.linspace(v['X'].min(), v['X'].max(), x_range)
        if param_model:
            ysp = []
            for x in xsp:
                xarg = [*k[:param_idx], x, *k[param_idx:]]
                ysp.append(param_model(state_or_trans, attribute, param = xarg))
            plt.plot(xsp, ysp, "r-", color=cm(i), linewidth=0.5)
            YY.append(ysp)
            YY_legend.append(legend_sanitizer.sub('_', 'regr_{}'.format(k)))
        if extra_function != None:
            ysp = []
            with np.errstate(divide='ignore', invalid='ignore'):
                for x in xsp:
                    xarg = [*k[:param_idx], x, *k[param_idx:]]
                    ysp.append(extra_function(*xarg))
            plt.plot(xsp, ysp, "r--", color=cm(i), linewidth=1, dashes=(3, 3))
            YY.append(ysp)
            YY_legend.append(legend_sanitizer.sub('_', 'symb_{}'.format(k)))

    with open(function_filename, 'w') as f:
        print(' '.join(YY_legend), file=f)
        for elem in np.array(YY).T:
            print(' '.join(map(str, elem)), file=f)

    print(data_filename_base, function_filename)
    if output:
        plt.savefig(output)
    else:
        plt.show()


def plot_param_fit(function, name, fitfunc, funp, parameters, datatype, index, X, Y, xaxis=None, yaxis=None):
    fig, ax1 = plt.subplots(figsize=(10,6))
    fig.canvas.set_window_title("fit %s" % (function))
    plt.subplots_adjust(left=0.14, right=0.99, top=0.99, bottom=0.14)

    x_range = X[index].max() - X[index].min() + 1

    if x_range > 100 and x_range < 500:
        xsp = np.linspace(X[index].min(), X[index].max(), x_range)
    else:
        xsp = np.linspace(X[index].min(), X[index].max(), 100)
        x_range = 100

    if xaxis != None:
        ax1.set_xlabel(xaxis)
    else:
        ax1.set_xlabel(parameters[index])
    if yaxis != None:
        ax1.set_ylabel(yaxis)
    else:
        ax1.set_ylabel('%s %s' % (name, datatype))

    otherparams = list(set(itertools.product(*X[:index], *X[index+1:])))
    cm = plt.get_cmap('brg', len(otherparams))
    for i in range(len(otherparams)):
        elem = otherparams[i]
        color = cm(i)

        tt = np.full((len(X[index])), True, dtype=bool)
        for k in range(len(parameters)):
            if k < index:
                tt &= X[k] == elem[k]
            elif k > index:
                tt &= X[k] == elem[k-1]

        plt.plot(X[index][tt], Y[tt], "rx", color=color)

        xarg = [np.array([x] * x_range) for x in elem[:index]]
        xarg.append(xsp)
        xarg.extend([np.array([x] * x_range) for x in elem[index:]])
        plt.plot(xsp, fitfunc(funp, xarg), "r-", color=color)
    plt.show()


def boxplot(ticks, modeldata, onlinedata, mimosadata, xlabel, ylabel):
    fig, ax1 = plt.subplots(figsize=(10,6))
    fig.canvas.set_window_title('DriverEval')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    bp = plt.boxplot(mimosadata, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                alpha=0.5)

    ax1.set_axisbelow(True)
    #ax1.set_title('DriverEval')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    numBoxes = len(mimosadata)

    xtickNames = plt.setp(ax1, xticklabels=ticks)
    plt.setp(xtickNames, rotation=0, fontsize=10)

    boxColors = ['darkkhaki', 'royalblue']
    medians = list(range(numBoxes))
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        # Alternate between Dark Khaki and Royal Blue
        k = i % 2
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        #ax1.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        plt.plot([np.average(med.get_xdata())], [np.average(mimosadata[i])],
                color='w', marker='*', markeredgecolor='k')
        if modeldata:
            plt.plot([np.average(med.get_xdata())], [modeldata[i]],
                color='w', marker='o', markeredgecolor='k')

    pos = np.arange(numBoxes) + 1
    upperLabels = [str(np.round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
        k = tick % 2
        y0, y1 = ax1.get_ylim()
        textpos = y0 + (y1 - y0)*0.97
        ypos = ax1.get_ylim()[0]
        ax1.text(pos[tick], textpos, upperLabels[tick],
                horizontalalignment='center', size='small',
                color='royalblue')

    plt.show()
