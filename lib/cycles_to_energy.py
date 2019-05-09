"""
Convert CPU cycle count to energy.

Contains classes for some embedded CPUs/MCUs. Given a configuration, each
class can convert a cycle count to an energy consumption.
"""

def get_class(cpu_name):
    """Return model class for cpu_name."""
    if cpu_name == 'MSP430':
        return MSP430
    if cpu_name == 'ATMega168':
        return ATMega168
    if cpu_name == 'ATTiny88':
        return ATTiny88

def _param_list_to_dict(device, param_list):
    param_dict = dict()
    for i, parameter in enumerate(sorted(device.parameters.keys())):
        param_dict[parameter] = param_list[i]
    return param_dict

class MSP430:
    name = 'MSP430'
    parameters = {
        'cpu_freq': [1e6, 4e6, 8e6, 12e6, 16e6],
        'memory' : ['unified', 'fram0', 'fram50', 'fram66', 'fram75', 'fram100', 'ram'],
        'voltage': [2.2, 3.0],
    }
    default_params = {
        'cpu_freq': 4e6,
        'memory' : 'unified',
        'voltage': 3
    }

    current_by_mem = {
        'unified' : [210,  640, 1220, 1475, 1845],
        'fram0'   : [370, 1280, 2510, 2080, 2650],
        'fram50'  : [240,  745, 1440, 1575, 1990],
        'fram66'  : [200,  560, 1070, 1300, 1620],
        'fram75'  : [170,  480,  890, 1155, 1420],
        'fram100' : [110,  235,  420,  640,  730],
        'ram'     : [130,  320,  585,  890, 1070],
    }

    def get_current(params):
        if type(params) != dict:
            return MSP430.get_current(_param_list_to_dict(MSP430, params))
        cpu_freq_index = MSP430.parameters['cpu_freq'].index(params['cpu_freq'])

        return MSP430.current_by_mem[params['memory']][cpu_freq_index] * 1e-6

    def get_power(params):
        if type(params) != dict:
            return MSP430.get_energy(_param_list_to_dict(MSP430, params))

        return MSP430.get_current(params) * params['voltage']

class ATMega168:
    name = 'ATMega168'
    parameters = {
        'cpu_freq': [1e6, 4e6, 8e6],
        'voltage': [2, 3, 5]
    }
    default_params = {
        'cpu_freq': 4e6,
        'voltage': 3
    }

    def get_current(params):
        if type(params) != dict:
            return ATMega168.get_current(_param_list_to_dict(ATMega168, params))
        if params['cpu_freq'] == 1e6 and params['voltage'] <= 2:
            return 0.5e-3
        if params['cpu_freq'] == 4e6 and params['voltage'] <= 3:
            return 3.5e-3
        if params['cpu_freq'] == 8e6 and params['voltage'] <= 5:
            return 12e-3
        return None

class ATTiny88:
    name = 'ATTiny88'
    parameters = {
        'cpu_freq': [1e6, 4e6, 8e6],
        'voltage': [2, 3, 5]
    }
    default_params = {
        'cpu_freq' : 4e6,
        'voltage' : 3
    }

    def get_current(params):
        if type(params) != dict:
            return ATTiny88.get_current(_param_list_to_dict(ATTiny88, params))
        if params['cpu_freq'] == 1e6 and params['voltage'] <= 2:
            return 0.2e-3
        if params['cpu_freq'] == 4e6 and params['voltage'] <= 3:
            return 1.4e-3
        if params['cpu_freq'] == 8e6 and params['voltage'] <= 5:
            return 4.5e-3
        return None

