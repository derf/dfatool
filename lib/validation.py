#!/usr/bin/env python3

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _xv_partitions_kfold(length, k=10):
    """
    Return k pairs of training and validation sets for k-fold cross-validation on `length` items.

    In k-fold cross-validation, every k-th item is used for validation and the remainder is used for training.
    As there are k ways to do this (items 0, k, 2k, ... vs. items 1, k+1, 2k+1, ... etc), this function returns k pairs of training and validation set.

    Note that this function operates on indices, not data.
    """
    pairs = []
    num_slices = k
    indexes = np.arange(length)
    for i in range(num_slices):
        training = np.delete(indexes, slice(i, None, num_slices))
        validation = indexes[i::num_slices]
        pairs.append((training, validation))
    return pairs


def _xv_param_partitions_kfold(param_values, k=10):
    indexes_by_param_value = dict()
    distinct_pv = list()
    for i, param_value in enumerate(param_values):
        pv = tuple(param_value)
        if pv in indexes_by_param_value:
            indexes_by_param_value[pv].append(i)
        else:
            distinct_pv.append(pv)
            indexes_by_param_value[pv] = [i]

    indexes = np.arange(len(distinct_pv))
    num_slices = k
    pairs = list()
    for i in range(num_slices):
        training_groups = np.delete(indexes, slice(i, None, num_slices))
        validation_groups = indexes[i::num_slices]
        training = list()
        for group in training_groups:
            training.extend(indexes_by_param_value[distinct_pv[group]])
        validation = list()
        for group in validation_groups:
            validation.extend(indexes_by_param_value[distinct_pv[group]])
        if not (len(training) and len(validation)):
            return None
        pairs.append((training, validation))
    return pairs


def _xv_partition_montecarlo(length):
    """
    Return training and validation set for Monte Carlo cross-validation on `length` items.

    This function operates on indices, not data. It randomly partitions range(length) into a list of training indices and a list of validation indices.

    The training set contains 2/3 of all indices; the validation set consits of the remaining 1/3.

    Example: 9 items -> training = [7, 3, 8, 0, 4, 2], validation = [ 1, 6, 5]
    """
    shuffled = np.random.permutation(np.arange(length))
    border = int(length * float(2) / 3)
    training = shuffled[:border]
    validation = shuffled[border:]
    return (training, validation)


class CrossValidator:
    """
    Cross-Validation helper for model generation.

    Given a set of measurements and a model class, it will partition the
    data into training and validation sets, train the model on the training
    set, and assess its quality on the validation set. This is repeated
    several times depending on cross-validation algorithm and configuration.
    Reports the mean model error over all cross-validation runs.
    """

    def __init__(self, model_class, by_name, parameters, *args, **kwargs):
        """
        Create a new CrossValidator object.

        Does not perform cross-validation yet.

        arguments:
        model_class -- model class/type used for model synthesis,
            e.g. PTAModel or AnalyticModel. model_class must have a
            constructor accepting (by_name, parameters, *args, **kwargs)
            and provide an `assess` method.
        by_name -- measurements aggregated by state/transition/function/... name.
            Layout: by_name[name][attribute] = list of data. Additionally,
            by_name[name]['attributes'] must be set to the list of attributes,
            e.g. ['power'] or ['duration', 'energy'].
        """
        self.model_class = model_class
        self.by_name = by_name
        self.names = sorted(by_name.keys())
        self.parameters = sorted(parameters)
        self.parameter_aware = False
        self.args = args
        self.kwargs = kwargs

    def kfold(self, model_getter, k=10):
        """
        Perform k-fold cross-validation and return average model quality.

        The by_name data is divided into 1-1/k training and 1/k validation in a deterministic manner.
        After creating a model for the training set, the
        model type returned by model_getter is evaluated on the validation set.
        This is repeated k times; the average of all measures is returned to the user.

        arguments:
        model_getter -- function with signature (model_object) -> model,
            e.g. lambda m: m.get_fitted()[0] to evaluate the parameter-aware
            model with automatic parameter detection.
        k -- step size for k-fold cross-validation. The validation set contains 100/k % of data.

        return value:
        dict of model quality measures.
        {
            'by_name' : {
                for each name: {
                    for each attribute: {
                        'mae' : mean of all mean absolute errors
                        'mae_list' : list of the individual MAE values encountered during cross-validation
                        'smape' : mean of all symmetric mean absolute percentage errors
                        'smape_list' : list of the individual SMAPE values encountered during cross-validation
                    }
                }
            }
        }
        """

        # training / validation subsets for each state and transition
        subsets_by_name = dict()
        training_and_validation_sets = list()

        for name in self.names:
            param_values = self.by_name[name]["param"]
            if self.parameter_aware:
                subsets_by_name[name] = _xv_param_partitions_kfold(param_values, k)
                if subsets_by_name[name] is None:
                    logger.warning(
                        f"Insufficient amount of parameter combinations for {name}, falling back to parameter-unaware cross-validation"
                    )
                    subsets_by_name[name] = _xv_partitions_kfold(len(param_values), k)
            else:
                subsets_by_name[name] = _xv_partitions_kfold(len(param_values), k)

        for i in range(k):
            training_and_validation_sets.append(dict())
            for name in self.names:
                training_and_validation_sets[i][name] = subsets_by_name[name][i]

        return self._generic_xv(model_getter, training_and_validation_sets)

    def montecarlo(self, model_getter, count=200):
        """
        Perform Monte Carlo cross-validation and return average model quality.

        The by_name data is randomly divided into 2/3 training and 1/3
        validation. After creating a model for the training set, the
        model type returned by model_getter is evaluated on the validation set.
        This is repeated count times (defaulting to 200); the average of all
        measures is returned to the user.

        arguments:
        model_getter -- function with signature (model_object) -> model,
            e.g. lambda m: m.get_fitted()[0] to evaluate the parameter-aware
            model with automatic parameter detection.
        count -- number of validation runs to perform, defaults to 200

        return value:
        dict of model quality measures.
        {
            'by_name' : {
                for each name: {
                    for each attribute: {
                        'mae' : mean of all mean absolute errors
                        'mae_list' : list of the individual MAE values encountered during cross-validation
                        'smape' : mean of all symmetric mean absolute percentage errors
                        'smape_list' : list of the individual SMAPE values encountered during cross-validation
                    }
                }
            }
        }
        """

        # training / validation subsets for each state and transition
        subsets_by_name = dict()
        training_and_validation_sets = list()

        for name in self.names:
            sample_count = len(self.by_name[name]["param"])
            subsets_by_name[name] = list()
            for _ in range(count):
                subsets_by_name[name].append(_xv_partition_montecarlo(sample_count))

        for i in range(count):
            training_and_validation_sets.append(dict())
            for name in self.names:
                training_and_validation_sets[i][name] = subsets_by_name[name][i]

        return self._generic_xv(model_getter, training_and_validation_sets)

    def _generic_xv(self, model_getter, training_and_validation_sets):
        ret = dict()

        for name in self.names:
            ret[name] = dict()
            for attribute in self.by_name[name]["attributes"]:
                ret[name][attribute] = {
                    "mae_list": list(),
                    "rmsd_list": list(),
                    "smape_list": list(),
                }

        for training_and_validation_by_name in training_and_validation_sets:
            res = self._single_xv(model_getter, training_and_validation_by_name)
            for name in self.names:
                for attribute in self.by_name[name]["attributes"]:
                    for measure in ("mae", "rmsd", "smape"):
                        ret[name][attribute][f"{measure}_list"].append(
                            res[name][attribute][measure]
                        )

        for name in self.names:
            for attribute in self.by_name[name]["attributes"]:
                for measure in ("mae", "rmsd", "smape"):
                    ret[name][attribute][measure] = np.mean(
                        ret[name][attribute][f"{measure}_list"]
                    )

        return ret

    def _single_xv(self, model_getter, tv_set_dict):
        training = dict()
        validation = dict()
        for name in self.names:
            training[name] = {"attributes": self.by_name[name]["attributes"]}
            validation[name] = {"attributes": self.by_name[name]["attributes"]}

            if "isa" in self.by_name[name]:
                training[name]["isa"] = self.by_name[name]["isa"]
                validation[name]["isa"] = self.by_name[name]["isa"]

            training_subset, validation_subset = tv_set_dict[name]

            for attribute in self.by_name[name]["attributes"]:
                self.by_name[name][attribute] = np.array(self.by_name[name][attribute])
                training[name][attribute] = self.by_name[name][attribute][
                    training_subset
                ]
                validation[name][attribute] = self.by_name[name][attribute][
                    validation_subset
                ]

            # We can't use slice syntax for 'param', which may contain strings and other odd values
            training[name]["param"] = list()
            validation[name]["param"] = list()
            for idx in training_subset:
                training[name]["param"].append(self.by_name[name]["param"][idx])
            for idx in validation_subset:
                validation[name]["param"].append(self.by_name[name]["param"][idx])

        training_data = self.model_class(
            training, self.parameters, *self.args, **self.kwargs
        )
        training_model = model_getter(training_data)
        validation_data = self.model_class(
            validation, self.parameters, *self.args, **self.kwargs
        )

        return validation_data.assess(training_model)
