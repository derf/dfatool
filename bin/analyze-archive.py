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
        print('{:10s}: {:.0f} µW'.format(state, static_model(state, 'power')))
    for trans in model.transitions():
        print('{:10s}: {:.0f} / {:.0f} / {:.0f} pJ'.format(
            trans, static_model(trans, 'energy'),
            static_model(trans, 'rel_energy_prev'),
            static_model(trans, 'rel_energy_next')))
        print('{:10s}: {:.0f} µs'.format(trans, static_model(trans, 'duration')))
    model.assess(static_model)

    print('--- LUT ---')
    lut_model = model.get_param_lut()
    model.assess(lut_model)

    sys.exit(0)
