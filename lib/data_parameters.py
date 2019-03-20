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

# TODO distinguish between int and uint, which is not visible from the
# data value alone
def _int_value_length(json):
    if type(json) == int:
        if json < 256:
            return 1
        if json < 65536:
            return 2
        return 4

    if type(json) == dict:
        return sum(map(_int_value_length, json.values()))

    if type(json) == list:
        return sum(map(_int_value_length, json))

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
    ret['bytelen_int'] = _int_value_length(json)
    ret['num_int'] = _num_of_type(json, int)
    ret['num_float'] = _num_of_type(json, float)
    ret['num_str'] = _num_of_type(json, str)

    return ret


class Protolog:
    """
    Loader and postprocessor for raw protobench (protocol-modeling/benchmark.py) data.

    Converts data sorted by (arch,lib)/benchmark/index/attribute
    to data sorted by (benchmark,index)/arch/lib/attribute.

    Once constructed, a class object provides three members:

    libraries -- array of library:config elements found in the benchmark results
    architectures -- array of multipass architecture names
    aggregate -- enriched log data, ordered by benchmark: {
        ('benchmark name', 'sub-benchmark index') : {
            'architecture' : {
                'library:options' : {
                    'attribute' : value (usually int or array)
                }
            }
        }
    }

    aggregate attributes:
    bss_{nop,ser,serdes} : whole-program Block Storage Segment (BSS) size
    callcycles_raw : { 'C++ statement' : [CPU cycles for execution] ... }.
        Not adjusted for 'nop' cycles -> values are a few cycles higher than true duration
    cycles_{ser,des,enc,dec,encser,desdec} : cycles for complete (de)serialization step,
        measured using just one counter start/stop (not a sum of callcycles_raw entries).
        Adjusted for 'nop' cycles -> should give accurate function call duration
    data_{secnop,ser,serdes} : whole-program Data Segment size
    heap_{ser,des} : Maximum heap usage during step
    serialized_size : Size (Bytes) of serialized data
    stack_alloc_{ser,des} : Maximum stack usage (Bytes) during step.
        Based on online analysis (comparison of memory dumps)
    stack_set_{ser,des} : Number of stack bytes modified during step.
        Based on online analysis (comparison of memory dumps), should be
        smaller than the corresponding stack_alloc_ value
    text_{nop,ser,serdes} : whole-program Text Segment (code/Flash) size
    """

    idem = lambda x: x
    datamap = [
        ['bss_nop', 'bss_size_nop', idem],
        ['bss_ser', 'bss_size_ser', idem],
        ['bss_serdes', 'bss_size_serdes', idem],
        ['callcycles_raw', 'callcycles', idem],
        ['cycles_ser', 'cycles', lambda x: max(0, int(np.mean(x['ser']) - np.mean(x['nop'])))],
        ['cycles_des', 'cycles', lambda x: max(0, int(np.mean(x['des']) - np.mean(x['nop'])))],
        ['cycles_enc', 'cycles', lambda x: max(0, int(np.mean(x['enc']) - np.mean(x['nop'])))],
        ['cycles_dec', 'cycles', lambda x: max(0, int(np.mean(x['dec']) - np.mean(x['nop'])))],
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
        """
        Load and enrich raw protobench log data.

        The enriched data can be accessed via the .aggregate class member,
        see the class documentation for details.
        """
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
                        except TypeError as e:
                            print('TypeError in {} {} {} {}: {}'.format(
                                arch_lib, benchmark, benchmark_item, aggregate_label,
                                str(e)))
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
        """
        Set self.aggregate[key][arch][lib][aggregate_Label] = getter(value[data_label]['v']).

        Additionally, add lib to self.libraries and arch to self.architectures
        key usually is ('benchmark name', 'sub-benchmark index').
        """
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
