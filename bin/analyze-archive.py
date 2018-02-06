#!/usr/bin/env python3

import sys
from dfatool import EnergyModel, RawData

if __name__ == '__main__':
    filenames = sys.argv[1:]
    raw_data = RawData(filenames)

    preprocessed_data = raw_data.get_preprocessed_data()
    model = EnergyModel(preprocessed_data)

    print('--- simple static model ---')
    static_model = model.get_static()
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
    model.assess(static_model)

    print('--- LUT ---')
    lut_model = model.get_param_lut()
    model.assess(lut_model)

    sys.exit(0)
