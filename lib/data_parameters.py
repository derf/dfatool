"""
Utilities for parameter extraction from data layout.

Parameters include the amount of keys, length of strings (both keys and values),
length of lists, ane more.
"""

import numpy as np
import ubjson

def _string_value_length(json):
    if type(json) == str:
        return len(json)

    if type(json) == dict:
        return sum(map(_string_value_length, json.values()))

    if type(json) == list:
        return sum(map(_string_value_length, json))

    return 0

def _string_key_length(json):
    if type(json) == dict:
        return sum(map(len, json.keys())) + sum(map(_string_key_length, json.values()))

    return 0

def _num_keys(json):
    if type(json) == dict:
        return len(json.keys()) + sum(map(_num_keys, json.values()))

    return 0

def _num_of_type(json, wanted_type):
    ret = 0
    if type(json) == wanted_type:
        ret = 1

    if type(json) == dict:
        ret += sum(map(lambda x: _num_of_type(x, wanted_type), json.values()))

    if type(json) == list:
        ret += sum(map(lambda x: _num_of_type(x, wanted_type), json))

    return ret

def json_to_param(json):
    """Return numeric parameters describing the structure of JSON data."""

    ret = dict()

    ret['strlen_keys'] = _string_key_length(json)
    ret['strlen_values'] = _string_value_length(json)
    #ret['num_keys'] = _num_keys(json)
    ret['num_int'] = _num_of_type(json, int)
    ret['num_float'] = _num_of_type(json, float)
    ret['num_str'] = _num_of_type(json, str)

    return ret


class Protolog:

    idem = lambda x: x
    datamap = [
        ['bss_nop', 'bss_size_nop', idem],
        ['bss_ser', 'bss_size_ser', idem],
        ['bss_serdes', 'bss_size_serdes', idem],
        ['cycles_ser', 'cycles', lambda x: int(np.mean(x['ser']) - np.mean(x['nop']))],
        ['cycles_des', 'cycles', lambda x: int(np.mean(x['des']) - np.mean(x['nop']))],
        ['cycles_enc', 'cycles', lambda x: int(np.mean(x['enc']) - np.mean(x['nop']))],
        ['cycles_dec', 'cycles', lambda x: int(np.mean(x['dec']) - np.mean(x['nop']))],
        ['cycles_encser', 'cycles', lambda x:
            int(np.mean(x['ser']) + np.mean(x['enc']) - 2 * np.mean(x['nop']))
        ],
        ['cycles_desdec', 'cycles', lambda x:
            int(np.mean(x['des']) + np.mean(x['dec']) - 2 * np.mean(x['nop']))
        ],
        ['cycles_ser_arr', 'cycles', lambda x: np.array(x['ser']) - np.mean(x['nop'])],
        ['cycles_des_arr', 'cycles', lambda x: np.array(x['des']) - np.mean(x['nop'])],
        ['cycles_enc_arr', 'cycles', lambda x: np.array(x['enc']) - np.mean(x['nop'])],
        ['cycles_dec_arr', 'cycles', lambda x: np.array(x['dec']) - np.mean(x['nop'])],
        ['data_nop', 'data_size_nop', idem],
        ['data_ser', 'data_size_ser', idem],
        ['data_serdes', 'data_size_serdes', idem],
        ['heap_ser', 'heap_usage_ser', idem],
        ['heap_des', 'heap_usage_des', idem],
        ['serialized_size', 'serialized_size', idem],
        ['stack_alloc_ser', 'stack_online_ser', lambda x: x['allocated']],
        ['stack_set_ser', 'stack_online_ser', lambda x: x['used']],
        ['stack_alloc_des', 'stack_online_des', lambda x: x['allocated']],
        ['stack_set_des', 'stack_online_des', lambda x: x['used']],
        ['text_nop', 'text_size_nop', idem],
        ['text_ser', 'text_size_ser', idem],
        ['text_serdes', 'text_size_serdes', idem],
    ]

    def __init__(self, logfile):
        with open(logfile, 'rb') as f:
            self.data = ubjson.load(f)
        self.libraries = set()
        self.architectures = set()
        self.aggregate = dict()

        for arch_lib in self.data.keys():
            arch, lib, libopts = arch_lib.split(':')
            library = lib + ':' + libopts
            for benchmark in self.data[arch_lib].keys():
                for benchmark_item in self.data[arch_lib][benchmark].keys():
                    subv = self.data[arch_lib][benchmark][benchmark_item]
                    for aggregate_label, data_label, getter in Protolog.datamap:
                        try:
                            self.add_datapoint(arch, library, (benchmark, benchmark_item), subv, aggregate_label, data_label, getter)
                        except KeyError:
                            pass

        for key in self.aggregate.keys():
            for arch in self.aggregate[key].keys():
                for lib, val in self.aggregate[key][arch].items():
                    try:
                        val['total_dmem_ser'] = val['stack_alloc_ser']
                        val['total_dmem_ser'] += val['heap_ser']
                    except KeyError:
                        pass
                    try:
                        val['total_dmem_des'] = val['stack_alloc_des']
                        val['total_dmem_des'] += val['heap_des']
                    except KeyError:
                        pass
                    try:
                        val['total_smem_ser'] = val['data_ser'] + val['bss_ser'] - val['data_nop'] - val['bss_nop']
                        val['total_smem_serdes'] = val['data_serdes'] + val['bss_serdes'] - val['data_nop'] - val['bss_nop']
                    except KeyError:
                        pass
                    try:
                        val['total_mem_ser'] = val['total_smem_ser'] + val['total_dmem_ser']
                    except KeyError:
                        pass
                    try:
                        val['text_serdes_delta'] = val['text_serdes'] - val['text_nop']
                    except KeyError:
                        pass
                    #try:
                    #    val['text_ser'] = val['text_nopser'] - val['text_nop']
                    #    val['text_des'] = val['text_nopserdes'] - val['text_nopser'] # use with care, probably bogus
                    #    val['text_serdes'] = val['text_nopserdes'] - val['text_nop']
                    #except KeyError:
                    #    pass

    def add_datapoint(self, arch, lib, key, value, aggregate_label, data_label, getter):
        if data_label in value and 'v' in value[data_label]:
            self.architectures.add(arch)
            self.libraries.add(lib)
            if not key in self.aggregate:
                self.aggregate[key] = dict()
            if not arch in self.aggregate[key]:
                self.aggregate[key][arch] = dict()
            if not lib in self.aggregate[key][arch]:
                self.aggregate[key][arch][lib] = dict()
            self.aggregate[key][arch][lib][aggregate_label] = getter(value[data_label]['v'])