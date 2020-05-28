"""
Utilities for parameter extraction from data layout.

Parameters include the amount of keys, length of strings (both keys and values),
length of lists, ane more.
"""

from .protocol_benchmarks import codegen_for_lib
from . import cycles_to_energy, size_to_radio_energy, utils
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

    ret["strlen_keys"] = _string_key_length(json)
    ret["strlen_values"] = _string_value_length(json)
    ret["bytelen_int"] = _int_value_length(json)
    ret["num_int"] = _num_of_type(json, int)
    ret["num_float"] = _num_of_type(json, float)
    ret["num_str"] = _num_of_type(json, str)

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

    def _median_cycles(data, key):
        # There should always be more than just one measurement -- otherwise
        # something went wrong
        if len(data[key]) <= 1:
            return np.nan
        for val in data[key]:
            # bogus data
            if val > 10_000_000:
                return np.nan
        for val in data["nop"]:
            # bogus data
            if val > 10_000_000:
                return np.nan
        # All measurements in data[key] cover the same instructions, so they
        # should be identical -> it's safe to take the median.
        # However, we leave out the first measurement as it is often bogus.
        if key == "nop":
            return np.median(data["nop"][1:])
        return max(0, int(np.median(data[key][1:]) - np.median(data["nop"][1:])))

    def _median_callcycles(data):
        ret = dict()
        for line in data.keys():
            ret[line] = np.median(data[line])
        return ret

    idem = lambda x: x
    datamap = [
        ["bss_nop", "bss_size_nop", idem],
        ["bss_ser", "bss_size_ser", idem],
        ["bss_serdes", "bss_size_serdes", idem],
        ["callcycles_raw", "callcycles", idem],
        ["callcycles_median", "callcycles", _median_callcycles],
        # Used to remove nop cycles from callcycles_median
        ["cycles_nop", "cycles", lambda x: Protolog._median_cycles(x, "nop")],
        ["cycles_ser", "cycles", lambda x: Protolog._median_cycles(x, "ser")],
        ["cycles_des", "cycles", lambda x: Protolog._median_cycles(x, "des")],
        ["cycles_enc", "cycles", lambda x: Protolog._median_cycles(x, "enc")],
        ["cycles_dec", "cycles", lambda x: Protolog._median_cycles(x, "dec")],
        # ['cycles_ser_arr', 'cycles', lambda x: np.array(x['ser'][1:]) - np.mean(x['nop'][1:])],
        # ['cycles_des_arr', 'cycles', lambda x: np.array(x['des'][1:]) - np.mean(x['nop'][1:])],
        # ['cycles_enc_arr', 'cycles', lambda x: np.array(x['enc'][1:]) - np.mean(x['nop'][1:])],
        # ['cycles_dec_arr', 'cycles', lambda x: np.array(x['dec'][1:]) - np.mean(x['nop'][1:])],
        ["data_nop", "data_size_nop", idem],
        ["data_ser", "data_size_ser", idem],
        ["data_serdes", "data_size_serdes", idem],
        ["heap_ser", "heap_usage_ser", idem],
        ["heap_des", "heap_usage_des", idem],
        ["serialized_size", "serialized_size", idem],
        ["stack_alloc_ser", "stack_online_ser", lambda x: x["allocated"]],
        ["stack_set_ser", "stack_online_ser", lambda x: x["used"]],
        ["stack_alloc_des", "stack_online_des", lambda x: x["allocated"]],
        ["stack_set_des", "stack_online_des", lambda x: x["used"]],
        ["text_nop", "text_size_nop", idem],
        ["text_ser", "text_size_ser", idem],
        ["text_serdes", "text_size_serdes", idem],
    ]

    def __init__(
        self,
        logfile,
        cpu_conf=None,
        cpu_conf_str=None,
        radio_conf=None,
        radio_conf_str=None,
    ):
        """
        Load and enrich raw protobench log data.

        The enriched data can be accessed via the .aggregate class member,
        see the class documentation for details.
        """
        self.cpu = None
        self.radio = None
        with open(logfile, "rb") as f:
            self.data = ubjson.load(f)
        self.libraries = set()
        self.architectures = set()
        self.aggregate = dict()

        for arch_lib in self.data.keys():
            arch, lib, libopts = arch_lib.split(":")
            library = lib + ":" + libopts
            for benchmark in self.data[arch_lib].keys():
                for benchmark_item in self.data[arch_lib][benchmark].keys():
                    subv = self.data[arch_lib][benchmark][benchmark_item]
                    for aggregate_label, data_label, getter in Protolog.datamap:
                        try:
                            self.add_datapoint(
                                arch,
                                library,
                                (benchmark, benchmark_item),
                                subv,
                                aggregate_label,
                                data_label,
                                getter,
                            )
                        except KeyError:
                            pass
                        except TypeError as e:
                            print(
                                "TypeError in {} {} {} {}: {} -> {}".format(
                                    arch_lib,
                                    benchmark,
                                    benchmark_item,
                                    aggregate_label,
                                    subv[data_label]["v"],
                                    str(e),
                                )
                            )
                            pass
                    try:
                        codegen = codegen_for_lib(lib, libopts.split(","), subv["data"])
                        if codegen.max_serialized_bytes != None:
                            self.add_datapoint(
                                arch,
                                library,
                                (benchmark, benchmark_item),
                                subv,
                                "buffer_size",
                                data_label,
                                lambda x: codegen.max_serialized_bytes,
                            )
                        else:
                            self.add_datapoint(
                                arch,
                                library,
                                (benchmark, benchmark_item),
                                subv,
                                "buffer_size",
                                data_label,
                                lambda x: 0,
                            )
                    except:
                        # avro's codegen will raise RuntimeError("Unsupported Schema") on unsupported data. Other libraries may just silently ignore it.
                        self.add_datapoint(
                            arch,
                            library,
                            (benchmark, benchmark_item),
                            subv,
                            "buffer_size",
                            data_label,
                            lambda x: 0,
                        )
                    # self.aggregate[(benchmark, benchmark_item)][arch][lib][aggregate_label] = getter(value[data_label]['v'])

        for key in self.aggregate.keys():
            for arch in self.aggregate[key].keys():
                for lib, val in self.aggregate[key][arch].items():
                    try:
                        val["cycles_encser"] = val["cycles_enc"] + val["cycles_ser"]
                    except KeyError:
                        pass
                    try:
                        val["cycles_desdec"] = val["cycles_des"] + val["cycles_dec"]
                    except KeyError:
                        pass
                    try:
                        for line in val["callcycles_median"].keys():
                            val["callcycles_median"][line] -= val["cycles_nop"]
                    except KeyError:
                        pass
                    try:
                        val["data_serdes_delta"] = val["data_serdes"] - val["data_nop"]
                    except KeyError:
                        pass
                    try:
                        val["data_serdes_delta_nobuf"] = (
                            val["data_serdes"] - val["data_nop"] - val["buffer_size"]
                        )
                    except KeyError:
                        pass
                    try:
                        val["bss_serdes_delta"] = val["bss_serdes"] - val["bss_nop"]
                    except KeyError:
                        pass
                    try:
                        val["bss_serdes_delta_nobuf"] = (
                            val["bss_serdes"] - val["bss_nop"] - val["buffer_size"]
                        )
                    except KeyError:
                        pass
                    try:
                        val["text_serdes_delta"] = val["text_serdes"] - val["text_nop"]
                    except KeyError:
                        pass
                    try:
                        val["total_dmem_ser"] = val["stack_alloc_ser"]
                        val["written_dmem_ser"] = val["stack_set_ser"]
                        val["total_dmem_ser"] += val["heap_ser"]
                        val["written_dmem_ser"] += val["heap_ser"]
                    except KeyError:
                        pass
                    try:
                        val["total_dmem_des"] = val["stack_alloc_des"]
                        val["written_dmem_des"] = val["stack_set_des"]
                        val["total_dmem_des"] += val["heap_des"]
                        val["written_dmem_des"] += val["heap_des"]
                    except KeyError:
                        pass
                    try:
                        val["total_dmem_serdes"] = max(
                            val["total_dmem_ser"], val["total_dmem_des"]
                        )
                    except KeyError:
                        pass
                    try:
                        val["text_ser_delta"] = val["text_ser"] - val["text_nop"]
                        val["text_serdes_delta"] = val["text_serdes"] - val["text_nop"]
                    except KeyError:
                        pass
                    try:
                        val["bss_ser_delta"] = val["bss_ser"] - val["bss_nop"]
                        val["bss_serdes_delta"] = val["bss_serdes"] - val["bss_nop"]
                    except KeyError:
                        pass
                    try:
                        val["data_ser_delta"] = val["data_ser"] - val["data_nop"]
                        val["data_serdes_delta"] = val["data_serdes"] - val["data_nop"]
                    except KeyError:
                        pass
                    try:
                        val["allmem_ser"] = (
                            val["text_ser"]
                            + val["data_ser"]
                            + val["bss_ser"]
                            + val["total_dmem_ser"]
                            - val["buffer_size"]
                        )
                        val["allmem_serdes"] = (
                            val["text_serdes"]
                            + val["data_serdes"]
                            + val["bss_serdes"]
                            + val["total_dmem_serdes"]
                            - val["buffer_size"]
                        )
                    except KeyError:
                        pass
                    try:
                        val["smem_serdes"] = (
                            val["text_serdes"]
                            + val["data_serdes"]
                            + val["bss_serdes"]
                            - val["buffer_size"]
                        )
                    except KeyError:
                        pass

        if cpu_conf_str:
            cpu_conf = utils.parse_conf_str(cpu_conf_str)

        if cpu_conf:
            self.cpu_conf = cpu_conf
            cpu = self.cpu = cycles_to_energy.get_class(cpu_conf["model"])
            for key, value in cpu.default_params.items():
                if not key in cpu_conf:
                    cpu_conf[key] = value
            for key in self.aggregate.keys():
                for arch in self.aggregate[key].keys():
                    for lib, val in self.aggregate[key][arch].items():
                        # All energy data is stored in nanojoules (nJ)
                        try:
                            val["energy_enc"] = int(
                                val["cycles_enc"]
                                * cpu.get_power(cpu_conf)
                                / cpu_conf["cpu_freq"]
                                * 1e9
                            )
                        except KeyError:
                            pass
                        except ValueError:
                            print(
                                "cycles_enc is NaN for {} -> {} -> {}".format(
                                    arch, lib, key
                                )
                            )
                        try:
                            val["energy_ser"] = int(
                                val["cycles_ser"]
                                * cpu.get_power(cpu_conf)
                                / cpu_conf["cpu_freq"]
                                * 1e9
                            )
                        except KeyError:
                            pass
                        except ValueError:
                            print(
                                "cycles_ser is NaN for {} -> {} -> {}".format(
                                    arch, lib, key
                                )
                            )
                        try:
                            val["energy_encser"] = int(
                                val["cycles_encser"]
                                * cpu.get_power(cpu_conf)
                                / cpu_conf["cpu_freq"]
                                * 1e9
                            )
                        except KeyError:
                            pass
                        except ValueError:
                            print(
                                "cycles_encser is NaN for {} -> {} -> {}".format(
                                    arch, lib, key
                                )
                            )
                        try:
                            val["energy_des"] = int(
                                val["cycles_des"]
                                * cpu.get_power(cpu_conf)
                                / cpu_conf["cpu_freq"]
                                * 1e9
                            )
                        except KeyError:
                            pass
                        except ValueError:
                            print(
                                "cycles_des is NaN for {} -> {} -> {}".format(
                                    arch, lib, key
                                )
                            )
                        try:
                            val["energy_dec"] = int(
                                val["cycles_dec"]
                                * cpu.get_power(cpu_conf)
                                / cpu_conf["cpu_freq"]
                                * 1e9
                            )
                        except KeyError:
                            pass
                        except ValueError:
                            print(
                                "cycles_dec is NaN for {} -> {} -> {}".format(
                                    arch, lib, key
                                )
                            )
                        try:
                            val["energy_desdec"] = int(
                                val["cycles_desdec"]
                                * cpu.get_power(cpu_conf)
                                / cpu_conf["cpu_freq"]
                                * 1e9
                            )
                        except KeyError:
                            pass
                        except ValueError:
                            print(
                                "cycles_desdec is NaN for {} -> {} -> {}".format(
                                    arch, lib, key
                                )
                            )

        if radio_conf_str:
            radio_conf = utils.parse_conf_str(radio_conf_str)

        if radio_conf:
            self.radio_conf = radio_conf
            radio = self.radio = size_to_radio_energy.get_class(radio_conf["model"])
            for key, value in radio.default_params.items():
                if not key in radio_conf:
                    radio_conf[key] = value
            for key in self.aggregate.keys():
                for arch in self.aggregate[key].keys():
                    for lib, val in self.aggregate[key][arch].items():
                        try:
                            radio_conf["txbytes"] = val["serialized_size"]
                            if radio_conf["txbytes"] > 0:
                                val["energy_tx"] = int(
                                    radio.get_energy(radio_conf) * 1e9
                                )
                            else:
                                val["energy_tx"] = 0
                            val["energy_encsertx"] = (
                                val["energy_encser"] + val["energy_tx"]
                            )
                            val["energy_desdecrx"] = (
                                val["energy_desdec"] + val["energy_tx"]
                            )
                        except KeyError:
                            pass

    def add_datapoint(self, arch, lib, key, value, aggregate_label, data_label, getter):
        """
        Set self.aggregate[key][arch][lib][aggregate_Label] = getter(value[data_label]['v']).

        Additionally, add lib to self.libraries and arch to self.architectures
        key usually is ('benchmark name', 'sub-benchmark index').
        """
        if data_label in value and "v" in value[data_label]:
            self.architectures.add(arch)
            self.libraries.add(lib)
            if not key in self.aggregate:
                self.aggregate[key] = dict()
            if not arch in self.aggregate[key]:
                self.aggregate[key][arch] = dict()
            if not lib in self.aggregate[key][arch]:
                self.aggregate[key][arch][lib] = dict()
            self.aggregate[key][arch][lib][aggregate_label] = getter(
                value[data_label]["v"]
            )
