#!/usr/bin/env python3

import sys
import numpy as np
from dfatool.dfatool import (
    RawData,
    pta_trace_to_aggregate,
)
from dfatool.model import PTAModel, regression_measures
from gplearn.genetic import SymbolicRegressor
from multiprocessing import Pool


def splitidx_srs(length):
    shuffled = np.random.permutation(np.arange(length))
    border = int(length * float(2) / 3)
    training = shuffled[:border]
    validation = shuffled[border:]
    return (training, validation)


def _gp_fit(arg):
    param = arg[0]
    X = arg[1]
    Y = arg[2]
    est_gp = SymbolicRegressor(
        population_size=param[0],
        generations=450,
        parsimony_coefficient=param[1],
        function_set=param[2].split(" "),
        const_range=(-param[3], param[3]),
    )

    training, validation = splitidx_srs(len(Y))
    X_train = X[training]
    Y_train = Y[training]
    X_validation = X[validation]
    Y_validation = Y[validation]

    try:
        est_gp.fit(X_train, Y_train)
        return (
            param,
            str(est_gp._program),
            est_gp._program.raw_fitness_,
            regression_measures(est_gp.predict(X_validation), Y_validation),
        )
    except Exception as e:
        return (param, "Exception: {}".format(str(e)), 999999999)


if __name__ == "__main__":
    population_size = [100, 500, 1000, 2000, 5000, 10000]
    parsimony_coefficient = [0.1, 0.5, 0.1, 1]
    function_set = ["add mul", "add mul sub div", "add mul sub div sqrt log inv"]
    const_lim = [100000, 50000, 10000, 1000, 500, 10, 1]
    filenames = sys.argv[4:]
    raw_data = RawData(filenames)

    preprocessed_data = raw_data.get_preprocessed_data()
    by_name, parameters, arg_count = pta_trace_to_aggregate(preprocessed_data)
    model = PTAModel(by_name, parameters, arg_count, traces=preprocessed_data)

    by_param = model.by_param

    state_or_tran = sys.argv[1]

    model_attribute = sys.argv[2]

    dimension = int(sys.argv[3])

    X = [[] for i in range(dimension)]
    Y = []

    for key, val in by_param.items():
        if key[0] == state_or_tran and len(key[1]) == dimension:
            Y.extend(val[model_attribute])
            for i in range(dimension):
                X[i].extend([float(key[1][i])] * len(val[model_attribute]))

    X = np.array(X)
    Y = np.array(Y)

    paramqueue = []

    for popsize in population_size:
        for coef in parsimony_coefficient:
            for fs in function_set:
                for cl in const_lim:
                    for i in range(10):
                        paramqueue.append(((popsize, coef, fs, cl), X.T, Y))

    with Pool() as pool:
        results = pool.map(_gp_fit, paramqueue)

    for res in sorted(results, key=lambda r: r[2]):
        print("{} {:.0f} ({:.0f})\n{}".format(res[0], res[3]["mae"], res[2], res[1]))
