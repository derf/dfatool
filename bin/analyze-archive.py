#!/usr/bin/env python3

import sys
from dfatool import EnergyModel, RawData

if __name__ == '__main__':
    filenames = sys.argv[1:]
    raw_data = RawData(filenames)

    preprocessed_data = raw_data.get_preprocessed_data()
    model = EnergyModel(preprocessed_data)

    #print('--- simple static model ---')
    #static_model = model.get_static()
    #for state in model.states():
    #    print('{:10s}: {:.0f} µW  ({:.2f})'.format(
    #        state,
    #        static_model(state, 'power'),
    #        model.generic_param_dependence_ratio(state, 'power')))
    #    for param in model.parameters():
    #        print('{:10s}  dependence on {:15s}: {:.2f}'.format(
    #            '',
    #            param,
    #            model.param_dependence_ratio(state, 'power', param)))
    #for trans in model.transitions():
    #    print('{:10s}: {:.0f} / {:.0f} / {:.0f} pJ  ({:.2f} / {:.2f} / {:.2f})'.format(
    #        trans, static_model(trans, 'energy'),
    #        static_model(trans, 'rel_energy_prev'),
    #        static_model(trans, 'rel_energy_next'),
    #        model.generic_param_dependence_ratio(trans, 'energy'),
    #        model.generic_param_dependence_ratio(trans, 'rel_energy_prev'),
    #        model.generic_param_dependence_ratio(trans, 'rel_energy_next')))
    #    print('{:10s}: {:.0f} µs'.format(trans, static_model(trans, 'duration')))
    #model.assess(static_model)

    #print('--- LUT ---')
    #lut_model = model.get_param_lut()
    #model.assess(lut_model)

    print('--- param model ---')
    param_model, param_info = model.get_fitted()
    for state in model.states():
        for attribute in ['power']:
            if param_info(state, attribute):
                print('{:10s}: {}'.format(state, param_info(state, attribute)['function']._model_str))
    for trans in model.transitions():
        for attribute in ['energy', 'rel_energy_prev', 'rel_energy_next', 'duration', 'timeout']:
            if param_info(trans, attribute):
                print('{:10s}: {:10s}: {}'.format(trans, attribute, param_info(trans, attribute)['function']._model_str))

    sys.exit(0)
