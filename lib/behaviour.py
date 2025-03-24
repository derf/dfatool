#!/usr/bin/env python3

import logging
from . import utils

logger = logging.getLogger(__name__)


class EventSequenceModel:
    def __init__(self, models):
        self.models = models

    def _event_normalizer(self, event):
        event_normalizer = lambda p: p
        if "/" in event:
            v1, v2 = event.split("/")
            if utils.is_numeric(v1):
                event = v2.strip()
                event_normalizer = lambda p: utils.soft_cast_float(v1) / p
            elif utils.is_numeric(v2):
                event = v1.strip()
                event_normalizer = lambda p: p / utils.soft_cast_float(v2)
            else:
                raise RuntimeError(f"Cannot parse '{event}'")
        return event, event_normalizer

    def eval_strs(self, events, aggregate="sum", aggregate_init=0, use_lut=False):
        for event in events:
            event, event_normalizer = self._event_normalizer(event)
            nn, param = event.split("(")
            name, action = nn.split(".")
            param_model = None
            ref_model = None

            for model in self.models:
                if name in model.names and action in model.attributes(name):
                    ref_model = model
                    if use_lut:
                        param_model = model.get_param_lut(allow_none=True)
                    else:
                        param_model, param_info = model.get_fitted()
                    break

            if param_model is None:
                raise RuntimeError(f"Did not find a model for {name}.{action}")

            param = param.removesuffix(")")
            if param == "":
                param = dict()
            else:
                param = utils.parse_conf_str(param)

            param_list = utils.param_dict_to_list(param, ref_model.parameters)

            if not use_lut and not param_info(name, action).is_predictable(param_list):
                logging.warning(
                    f"Cannot predict {name}.{action}({param}), falling back to static model"
                )

            try:
                event_output = event_normalizer(
                    param_model(
                        name,
                        action,
                        param=param_list,
                    )
                )
            except KeyError:
                logging.error(f"Cannot predict {name}.{action}({param}) from LUT model")
                raise

            if aggregate == "sum":
                aggregate_init += event_output
            else:
                raise RuntimeError(f"Unknown aggregate type: {aggregate}")

        return aggregate_init
