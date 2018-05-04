#!/usr/bin/env python3

import getopt
import json
import plotter
import re
import sys
from dfatool import EnergyModel, RawData
from dfatool import soft_cast_int, is_numeric, gplearn_to_function

opts = {}

def print_model_quality(results):
    for state_or_tran in results.keys():
        print()
        for key, result in results[state_or_tran].items():
            if 'smape' in result:
                print('{:20s} {:15s} {:.2f}% / {:.0f}'.format(
                    state_or_tran, key, result['smape'], result['mae']))
            else:
                print('{:20s} {:15s} {:.0f}'.format(
                    state_or_tran, key, result['mae']))

def format_quality_measures(result):
    if 'smape' in result:
        return '{:6.2f}% / {:9.0f}'.format(result['smape'], result['mae'])
    else:
        return '{:6}    {:9.0f}'.format('', result['mae'])

def model_quality_table(result_lists, info_list):
    for state_or_tran in result_lists[0]['by_dfa_component'].keys():
        for key in result_lists[0]['by_dfa_component'][state_or_tran].keys():
            buf = '{:20s} {:15s}'.format(state_or_tran, key)
            for i, results in enumerate(result_lists):
                info = info_list[i]
                buf += '  |||  '
                if info == None or info(state_or_tran, key):
                    result = results['by_dfa_component'][state_or_tran][key]
                    buf += format_quality_measures(result)
                else:
                    buf += '{:6}----{:9}'.format('', '')
            print(buf)

def model_summary_table(result_list):
    buf = 'transition duration'
    for results in result_list:
        if len(buf):
            buf += '  |||  '
        buf += format_quality_measures(results['duration_by_trace'])
    print(buf)
    buf = 'total energy       '
    for results in result_list:
        if len(buf):
            buf += '  |||  '
        buf += format_quality_measures(results['energy_by_trace'])
    print(buf)
    buf = 'rel total energy   '
    for results in result_list:
        if len(buf):
            buf += '  |||  '
        buf += format_quality_measures(results['rel_energy_by_trace'])
    print(buf)
    buf = 'state-only energy  '
    for results in result_list:
        if len(buf):
            buf += '  |||  '
        buf += format_quality_measures(results['state_energy_by_trace'])
    print(buf)
    buf = 'transition timeout '
    for results in result_list:
        if len(buf):
            buf += '  |||  '
        buf += format_quality_measures(results['timeout_by_trace'])
    print(buf)


def print_text_model_data(model, pm, pq, lm, lq, am, ai, aq):
    print('')
    print(r'key attribute $1 - \frac{\sigma_X}{...}$')
    for state_or_tran in model.by_name.keys():
        for attribute in model.attributes(state_or_tran):
            print('{} {} {:.8f}'.format(state_or_tran, attribute, model.generic_param_dependence_ratio(state_or_tran, attribute)))

    print('')
    print(r'key attribute parameter $1 - \frac{...}{...}$')
    for state_or_tran in model.by_name.keys():
        for attribute in model.attributes(state_or_tran):
            for param in model.parameters():
                print('{} {} {} {:.8f}'.format(state_or_tran, attribute, param, model.param_dependence_ratio(state_or_tran, attribute, param)))
            if state_or_tran in model._num_args:
                for arg_index in range(model._num_args[state_or_tran]):
                    print('{} {} {:d} {:.8f}'.format(state_or_tran, attribute, arg_index, model.arg_dependence_ratio(state_or_tran, attribute, arg_index)))

if __name__ == '__main__':

    ignored_trace_indexes = None
    discard_outliers = None
    safe_functions_enabled = False
    function_override = {}
    show_models = []
    show_quality = []
    hwmodel = None
    energymodel_export_file = None

    try:
        optspec = (
            'plot-unparam= plot-param= show-models= show-quality= '
            'ignored-trace-indexes= discard-outliers= function-override= '
            'with-safe-functions hwmodel= export-energymodel='
        )
        raw_opts, args = getopt.getopt(sys.argv[1:], "", optspec.split(' '))

        for option, parameter in raw_opts:
            optname = re.sub(r'^--', '', option)
            opts[optname] = parameter

            if 'ignored-trace-indexes' in opts:
                ignored_trace_indexes = list(map(int, opts['ignored-trace-indexes'].split(',')))
                if 0 in ignored_trace_indexes:
                    print('[E] arguments to --ignored-trace-indexes start from 1')

            if 'discard-outliers' in opts:
                discard_outliers = float(opts['discard-outliers'])

            if 'function-override' in opts:
                for function_desc in opts['function-override'].split(';'):
                    state_or_tran, attribute, *function_str = function_desc.split(' ')
                    function_override[(state_or_tran, attribute)] = ' '.join(function_str)

            if 'show-models' in opts:
                show_models = opts['show-models'].split(',')

            if 'show-quality' in opts:
                show_quality = opts['show-quality'].split(',')

            if 'with-safe-functions' in opts:
                safe_functions_enabled = True

            if 'hwmodel' in opts:
                with open(opts['hwmodel'], 'r') as f:
                    hwmodel = json.load(f)

    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    raw_data = RawData(args)

    preprocessed_data = raw_data.get_preprocessed_data()
    model = EnergyModel(preprocessed_data,
        ignore_trace_indexes = ignored_trace_indexes,
        discard_outliers = discard_outliers,
        function_override = function_override,
        hwmodel = hwmodel)


    if 'plot-unparam' in opts:
        for kv in opts['plot-unparam'].split(';'):
            state_or_trans, attribute = kv.split(' ')
            plotter.plot_y(model.by_name[state_or_trans][attribute])

    if len(show_models):
        print('--- simple static model ---')
    static_model = model.get_static()
    if 'static' in show_models or 'all' in show_models:
        for state in model.states():
            print('{:10s}: {:.0f} µW  ({:.2f})'.format(
                state,
                static_model(state, 'power'),
                model.generic_param_dependence_ratio(state, 'power')))
            for param in model.parameters():
                print('{:10s}  dependence on {:15s}: {:.2f}'.format(
                    '',
                    param,
                    model.param_dependence_ratio(state, 'power', param)))
        for trans in model.transitions():
            print('{:10s}: {:.0f} / {:.0f} / {:.0f} pJ  ({:.2f} / {:.2f} / {:.2f})'.format(
                trans, static_model(trans, 'energy'),
                static_model(trans, 'rel_energy_prev'),
                static_model(trans, 'rel_energy_next'),
                model.generic_param_dependence_ratio(trans, 'energy'),
                model.generic_param_dependence_ratio(trans, 'rel_energy_prev'),
                model.generic_param_dependence_ratio(trans, 'rel_energy_next')))
            print('{:10s}: {:.0f} µs'.format(trans, static_model(trans, 'duration')))
    static_quality = model.assess(static_model)

    if len(show_models):
        print('--- LUT ---')
    lut_model = model.get_param_lut()
    lut_quality = model.assess(lut_model)

    if len(show_models):
        print('--- param model ---')
    param_model, param_info = model.get_fitted(safe_functions_enabled = safe_functions_enabled)
    if 'param' in show_models or 'all' in show_models:
        for state in model.states():
            for attribute in model.attributes(state):
                if param_info(state, attribute):
                    print('{:10s}: {}'.format(state, param_info(state, attribute)['function']._model_str))
                    print('{:10s}  {}'.format('', param_info(state, attribute)['function']._regression_args))
        for trans in model.attributes(trans):
            for attribute in ['energy', 'rel_energy_prev', 'rel_energy_next', 'duration', 'timeout']:
                if param_info(trans, attribute):
                    print('{:10s}: {:10s}: {}'.format(trans, attribute, param_info(trans, attribute)['function']._model_str))
                    print('{:10s}  {:10s}  {}'.format('', '', param_info(trans, attribute)['function']._regression_args))
    analytic_quality = model.assess(param_model)

    if 'tex' in show_models or 'tex' in show_quality:
        print_text_model_data(model, static_model, static_quality, lut_model, lut_quality, param_model, param_info, analytic_quality)

    if 'table' in show_quality or 'all' in show_quality:
        model_quality_table([static_quality, analytic_quality, lut_quality], [None, param_info, None])
    if 'summary' in show_quality or 'all' in show_quality:
        model_summary_table([static_quality, analytic_quality, lut_quality])

    if 'plot-param' in opts:
        for kv in opts['plot-param'].split(';'):
            state_or_trans, attribute, param_name, *function = kv.split(' ')
            if len(function):
                function = gplearn_to_function(' '.join(function))
            else:
                function = None
            plotter.plot_param(model, state_or_trans, attribute, model.param_index(param_name), extra_function=function)

    if 'export-energymodel' in opts:
        if not hwmodel:
            print('[E] --export-energymodel requires --hwmodel to be set')
            sys.exit(1)
        json_model = model.to_json()
        with open(opts['export-energymodel'], 'w') as f:
            json.dump(json_model, f, indent = 2, sort_keys = True)


    sys.exit(0)
