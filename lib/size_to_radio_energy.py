"""
Convert data length to radio TX/RX energy.

Contains classes for some embedded CPUs/MCUs. Given a configuration, each
class can convert a cycle count to an energy consumption.
"""

import numpy as np

def get_class(radio_name: str):
    """Return model class for radio_name."""
    if radio_name == 'CC1200tx':
        return CC1200tx

def _param_list_to_dict(device, param_list):
    param_dict = dict()
    for i, parameter in enumerate(sorted(device.parameters.keys())):
        param_dict[parameter] = param_list[i]
    return param_dict

class CC1200tx:
    name = 'CC1200tx'
    parameters = {
        'symbolrate' : [6, 12, 25, 50, 100, 200, 250], # ksps
        'txbytes' : [],
        'txpower' : [10, 20, 30, 40, 47], # dBm = f(txpower)
    }
    default_params = {
        'symbolrate' : 100,
        'txpower' : 47,
    }

    def get_energy(params):
        if type(params) != dict:
            return CC1200tx.get_energy(_param_list_to_dict(CC1200tx, params))

        power  = 8.18053941e+04
        power -= 1.24208376e+03 * np.sqrt(params['symbolrate'])
        power -= 5.73742779e+02 * np.log(params['txbytes'])
        power += 1.76945886e+01 * (params['txpower'])**2
        power += 2.33469617e+02 * np.sqrt(params['symbolrate']) * np.log(params['txbytes'])
        power -= 6.99137635e-01 * np.sqrt(params['symbolrate']) * (params['txpower'])**2
        power -= 3.31365158e-01 * np.log(params['txbytes']) * (params['txpower'])**2
        power += 1.32784945e-01 * np.sqrt(params['symbolrate']) * np.log(params['txbytes']) * (params['txpower'])**2

        duration  = 3.65513500e+02
        duration += 8.01016526e+04 * 1/(params['symbolrate'])
        duration -= 7.06364515e-03 * params['txbytes']
        duration += 8.00029860e+03 * 1/(params['symbolrate']) * params['txbytes']

        return power * 1e-6 * duration * 1e-6
