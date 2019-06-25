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
    if radio_name == 'CC1200rx':
        return CC1200rx
    if radio_name == 'NRF24L01tx':
        return NRF24L01tx
    if radio_name == 'NRF24L01dtx':
        return NRF24L01dtx
    if radio_name == 'esp8266dtx':
        return ESP8266dtx
    if radio_name == 'esp8266drx':
        return ESP8266drx

def _param_list_to_dict(device, param_list):
    param_dict = dict()
    for i, parameter in enumerate(sorted(device.parameters.keys())):
        param_dict[parameter] = param_list[i]
    return param_dict

class CC1200tx:
    """CC1200 TX energy based on aemr measurements."""
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

    @staticmethod
    def get_energy(params: dict):
        if type(params) != dict:
            return CC1200tx.get_energy(_param_list_to_dict(CC1200tx, params))

        # Mittlere TX-Leistung, gefitted von AEMR
        # Messdaten erhoben bei 3.6V
        power  = 8.18053941e+04
        power -= 1.24208376e+03 * np.sqrt(params['symbolrate'])
        power -= 5.73742779e+02 * np.log(params['txbytes'])
        power += 1.76945886e+01 * (params['txpower'])**2
        power += 2.33469617e+02 * np.sqrt(params['symbolrate']) * np.log(params['txbytes'])
        power -= 6.99137635e-01 * np.sqrt(params['symbolrate']) * (params['txpower'])**2
        power -= 3.31365158e-01 * np.log(params['txbytes']) * (params['txpower'])**2
        power += 1.32784945e-01 * np.sqrt(params['symbolrate']) * np.log(params['txbytes']) * (params['txpower'])**2

        # txDone-Timeout, gefitted von AEMR
        duration  = 3.65513500e+02
        duration += 8.01016526e+04 * 1/(params['symbolrate'])
        duration -= 7.06364515e-03 * params['txbytes']
        duration += 8.00029860e+03 * 1/(params['symbolrate']) * params['txbytes']

        # TX-Energie, gefitted von AEMR
        # Achtung: Energy ist in µJ, nicht (wie in AEMR-Transitionsmodellen üblich) in pJ
        # Messdaten erhoben bei 3.6V

        energy  = 1.74383259e+01
        energy += 6.29922138e+03 * 1/(params['symbolrate'])
        energy += 1.13307135e-02 * params['txbytes']
        energy -= 1.28121377e-04 * (params['txpower'])**2
        energy += 6.29080184e+02 * 1/(params['symbolrate']) * params['txbytes']
        energy += 1.25647926e+00 * 1/(params['symbolrate']) * (params['txpower'])**2
        energy += 1.31996202e-05 * params['txbytes'] * (params['txpower'])**2
        energy += 1.25676966e-01 * 1/(params['symbolrate']) * params['txbytes'] * (params['txpower'])**2

        return energy * 1e-6

    @staticmethod
    def get_energy_per_byte(params):
        A  = 8.18053941e+04
        A -= 1.24208376e+03 * np.sqrt(params['symbolrate'])
        A += 1.76945886e+01 * (params['txpower'])**2
        A -= 6.99137635e-01 * np.sqrt(params['symbolrate']) * (params['txpower'])**2
        B  = -5.73742779e+02
        B += 2.33469617e+02 * np.sqrt(params['symbolrate'])
        B -= 3.31365158e-01 * (params['txpower'])**2
        B += 1.32784945e-01 * np.sqrt(params['symbolrate']) * (params['txpower'])**2
        C  = 3.65513500e+02
        C += 8.01016526e+04 * 1/(params['symbolrate'])
        D  = -7.06364515e-03
        D += 8.00029860e+03 * 1/(params['symbolrate'])

        x = params['txbytes']

        # in pJ
        de_dx = A * D + B * C * 1/x + B * D * (np.log(x) + 1)

        # in µJ
        de_dx  = 1.13307135e-02
        de_dx += 6.29080184e+02 * 1/(params['symbolrate'])
        de_dx += 1.31996202e-05 * (params['txpower'])**2
        de_dx += 1.25676966e-01 * 1/(params['symbolrate']) * (params['txpower'])**2

        #de_dx = (B * 1/x) * (C + D * x) + (A + B * np.log(x)) * D

        return de_dx * 1e-6

class CC1200rx:
    """CC1200 RX energy based on aemr measurements."""
    name = 'CC1200rx'
    parameters = {
        'symbolrate' : [6, 12, 25, 50, 100, 200, 250], # ksps
        'txbytes' : [],
        'txpower' : [10, 20, 30, 40, 47], # dBm = f(txpower)
    }
    default_params = {
        'symbolrate' : 100,
        'txpower' : 47,
    }

    @staticmethod
    def get_energy(params):
        # TODO
        return params['txbytes'] * CC1200rx.get_energy_per_byte(params)

    @staticmethod
    def get_energy_per_byte(params):
        #RX        : 0 + regression_arg(0) + regression_arg(1) * np.log(parameter(symbolrate) + 1)
        # [84414.91636169   205.63323036]

        de_dx = (84414.91636169 + 205.63323036 * np.log(params['symbolrate'] + 1)) * 8000 / params['symbolrate']

        return de_dx * 1e-12

class NRF24L01rx:
    """NRF24L01+ RX energy based on aemr measurements (using variable packet size)"""
    name = 'NRF24L01'
    parameters = {
        'datarate' : [250, 1000, 2000], # kbps
        'txbytes' : [],
        'txpower' : [-18, -12, -6, 0], # dBm
        'voltage' : [1.9, 3.6],
    }
    default_params = {
        'datarate' : 1000,
        'txpower' : -6,
        'voltage' : 3,
    }

    @staticmethod
    def get_energy_per_byte(params):
        # RX        : 0 + regression_arg(0) + regression_arg(1) * np.sqrt(parameter(datarate))
        #     [48530.73235537   117.25274402]

        de_dx = (48530.73235537 + 117.25274402 * np.sqrt(params['datarate'])) * 8000 / params['datarate']

        return de_dx * 1e-12

# PYTHONPATH=lib bin/analyze-archive.py --show-model=all --show-quality=table ../data/*_RF24_no_retries.tar
class NRF24L01tx:
    """NRF24L01+ TX(*) energy based on aemr measurements (32B fixed packet size, (*)ack-await, no retries)."""
    name = 'NRF24L01'
    parameters = {
        'datarate' : [250, 1000, 2000], # kbps
        'txbytes' : [],
        'txpower' : [-18, -12, -6, 0], # dBm
        'voltage' : [1.9, 3.6],
    }
    default_params = {
        'datarate' : 1000,
        'txpower' : -6,
        'voltage' : 3,
    }

# AEMR:
# TX power / energy:
#TX        : 0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate)) + regression_arg(2) * (19.47+parameter(txpower))**2 + regression_arg(3) * 1/(parameter(datarate)) * (19.47+parameter(txpower))**2
#            [6.30323056e+03 2.59889924e+06 7.82186268e+00 8.69746093e+03]
#TX        : 0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate)) + regression_arg(2) * (19.47+parameter(txpower))**2 + regression_arg(3) * 1/(parameter(datarate)) * (19.47+parameter(txpower))**2
#            [7.67932887e+00 1.02969455e+04 4.55116475e-03 2.99786534e+01]
#epilogue  : timeout   : 0 + regression_arg(0) + regression_arg(1) * 1/(parameter(datarate))
#            [  1624.06589147 332251.93798766]


    @staticmethod
    def get_energy(params):
        if type(params) != dict:
            return NRF24L01tx.get_energy(_param_list_to_dict(NRF24L01tx, params))

        # TX-Leistung, gefitted von AEMR
        # Messdaten erhoben bei 3.6V
        power = 6.30323056e+03
        power += 2.59889924e+06 * 1/params['datarate']
        power += 7.82186268e+00 * (19.47+params['txpower'])**2
        power += 8.69746093e+03 * 1/params['datarate'] *  (19.47+params['txpower'])**2

        # TX-Dauer, gefitted von AEMR
        duration = 1624.06589147
        duration += 332251.93798766 * 1/params['datarate']

        # TX-Energie, gefitted von AEMR
        # Achtung: Energy ist in µJ, nicht (wie in AEMR-Transitionsmodellen üblich) in pJ
        # Messdaten erhoben bei 3.6V
        energy = 7.67932887e+00
        energy += 1.02969455e+04 * 1/params['datarate']
        energy += 4.55116475e-03 * (19.47+params['txpower'])**2
        energy += 2.99786534e+01 * 1/params['datarate'] * (19.47+params['txpower'])**2

        energy = power * 1e-6 * duration * 1e-6 * np.ceil(params['txbytes'] / 32)

        return energy

    @staticmethod
    def get_energy_per_byte(params):
        if type(params) != dict:
            return NRF24L01tx.get_energy_per_byte(_param_list_to_dict(NRF24L01tx, params))

        # in µJ
        de_dx = 0


class NRF24L01dtx:
    """nRF24L01+ TX energy based on datasheet values (probably unerestimated)"""
    name = 'NRF24L01'
    parameters = {
        'datarate' : [250, 1000, 2000], # kbps
        'txbytes' : [],
        'txpower' : [-18, -12, -6, 0], # dBm
        'voltage' : [1.9, 3.6],
    }
    default_params = {
        'datarate' : 1000,
        'txpower' : -6,
        'voltage' : 3,
    }

    # 130 us RX settling: 8.9 mE
    # 130 us TX settling: 8 mA

    @staticmethod
    def get_energy(params):
        if type(params) != dict:
            return NRF24L01dtx.get_energy(_param_list_to_dict(NRF24L01dtx, params))

        header_bytes = 7

        # TX settling: 130 us @ 8 mA
        energy = 8e-3 * params['voltage'] * 130e-6

        if params['txpower'] == -18:
            current = 7e-3
        elif params['txpower'] == -12:
            current = 7.5e-3
        elif params['txpower'] == -6:
            current = 9e-3
        elif params['txpower'] == 0:
            current = 11.3e-3

        energy += current * params['voltage'] * ((header_bytes + params['txbytes']) * 8 / (params['datarate'] * 1e3))

        return energy

class ESP8266dtx:
    """esp8266 TX energy based on (hardly documented) datasheet values"""
    name = 'esp8266'
    parameters = {
        'voltage' : [2.5, 3.0, 3.3, 3.6],
        'txbytes' : [],
        'bitrate' : [65e6],
        'tx_current' : [120e-3],
    }
    default_params = {
        'voltage' : 3,
        'bitrate' : 65e6,
        'tx_current' : 120e-3
    }

    @staticmethod
    def get_energy(params):
        # TODO
        return 0

    @staticmethod
    def get_energy_per_byte(params):
        if type(params) != dict:
            return ESP8266dtx.get_energy_per_byte(_param_list_to_dict(ESP8266dtx, params))

        # TX in 802.11n MCS7 -> 64QAM, 65/72.2 Mbit/s @ 20MHz channel, 135/150 Mbit/s @ 40MHz
        # -> Value for 65 Mbit/s @ 20MHz channel
        return params['tx_current'] * params['voltage'] / params['bitrate']

class ESP8266drx:
    """esp8266 RX energy based on (hardly documented) datasheet values"""
    name = 'esp8266'
    parameters = {
        'voltage' : [2.5, 3.0, 3.3, 3.6],
        'txbytes' : [],
        'bitrate' : [65e6],
        'rx_current' : [56e-3],
    }
    default_params = {
        'voltage' : 3,
        'bitrate' : 65e6,
        'rx_current' : 56e-3
    }

    @staticmethod
    def get_energy(params):
        # TODO
        return 0

    @staticmethod
    def get_energy_per_byte(params):
        if type(params) != dict:
            return ESP8266drx.get_energy_per_byte(_param_list_to_dict(ESP8266drx, params))

        # TX in 802.11n MCS7 -> 64QAM, 65/72.2 Mbit/s @ 20MHz channel, 135/150 Mbit/s @ 40MHz
        # -> Value for 65 Mbit/s @ 20MHz channel
        return params['rx_current'] * params['voltage'] / params['bitrate']
