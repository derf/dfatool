#!/usr/bin/env python3

import logging
from . import utils
from .model import AnalyticModel
from . import functions as df

logger = logging.getLogger(__name__)


class SDKBehaviourModel:

    def __init__(self, observations, annotations):

        meta_observations = list()
        delta_by_name = dict()
        delta_param_by_name = dict()
        is_loop = dict()

        for annotation in annotations:
            # annotation.start.param may be incomplete, for instance in cases
            # where DPUs are allocated before the input file is loadeed (and
            # thus before the problem size is known).
            # Hence, we must use annotation.end.param whenever we deal
            # with possibly problem size-dependent behaviour.
            am_tt_param_names = sorted(annotation.end.param.keys())
            if annotation.name not in delta_by_name:
                delta_by_name[annotation.name] = dict()
                delta_param_by_name[annotation.name] = dict()
            _, _, meta_obs, _is_loop = self.learn_pta(
                observations,
                annotation,
                delta_by_name[annotation.name],
                delta_param_by_name[annotation.name],
            )
            meta_observations += meta_obs
            is_loop.update(_is_loop)

        self.am_tt_param_names = am_tt_param_names
        self.delta_by_name = delta_by_name
        self.delta_param_by_name = delta_param_by_name
        self.meta_observations = meta_observations
        self.is_loop = is_loop

        self.build_transition_guards()

    def build_transition_guards(self):
        self.transition_guard = dict()
        for name in sorted(self.delta_by_name.keys()):
            for t_from, t_to_set in self.delta_by_name[name].items():
                i_to_transition = dict()
                delta_param_sets = list()
                to_names = list()
                transition_guard = dict()

                if len(t_to_set) > 1:
                    am_tt_by_name = {
                        name: {
                            "attributes": [t_from],
                            "param": list(),
                            t_from: list(),
                        },
                    }
                    for i, t_to in enumerate(sorted(t_to_set)):
                        for param in self.delta_param_by_name[name][(t_from, t_to)]:
                            am_tt_by_name[name]["param"].append(
                                utils.param_dict_to_list(
                                    utils.param_str_to_dict(param),
                                    self.am_tt_param_names,
                                )
                            )
                            am_tt_by_name[name][t_from].append(i)
                            i_to_transition[i] = t_to
                    am = AnalyticModel(
                        am_tt_by_name, self.am_tt_param_names, force_tree=True
                    )
                    model, info = am.get_fitted()
                    if type(info(name, t_from)) is df.SplitFunction:
                        flat_model = info(name, t_from).flatten()
                    else:
                        flat_model = list()
                        logger.warning(
                            f"Model for {name} {t_from} is {info(name, t_from)}, expected SplitFunction"
                        )

                    for prefix, output in flat_model:
                        transition_name = i_to_transition[int(output)]
                        if transition_name not in transition_guard:
                            transition_guard[transition_name] = list()
                        transition_guard[transition_name].append(prefix)

                self.transition_guard[t_from] = transition_guard

    def get_trace(self, name, param_dict):
        delta = self.delta_by_name[name]
        current_state = "__init__"
        trace = [current_state]
        states_seen = set()
        while current_state != "__end__":
            next_states = delta[current_state]

            states_seen.add(current_state)
            next_states = list(filter(lambda q: q not in states_seen, next_states))

            if len(next_states) == 0:
                raise RuntimeError(
                    f"get_trace({name}, {param_dict}): found infinite loop at {trace}"
                )

            if len(next_states) > 1 and self.transition_guard[current_state]:
                matching_next_states = list()
                for candidate in next_states:
                    for condition in self.transition_guard[current_state][candidate]:
                        valid = True
                        for key, value in condition:
                            if param_dict[key] != value:
                                valid = False
                                break
                        if valid:
                            matching_next_states.append(candidate)
                            break
                next_states = matching_next_states

            if len(next_states) == 0:
                raise RuntimeError(
                    f"get_trace({name}, {param_dict}): found no valid outbound transitions at {trace}, candidates {self.transition_guard[current_state]}"
                )
            if len(next_states) > 1:
                raise RuntimeError(
                    f"get_trace({name}, {param_dict}): found non-deterministic outbound transitions {next_states} at {trace}"
                )

            (next_state,) = next_states

            trace.append(next_state)
            current_state = next_state

        return trace

    def learn_pta(self, observations, annotation, delta=dict(), delta_param=dict()):
        prev_i = annotation.start.offset
        prev = "__init__"
        prev_non_kernel = prev
        meta_observations = list()
        n_seen = dict()

        total_latency_us = 0

        if annotation.kernels:
            # ggf. als dict of tuples, für den Fall dass Schleifen verschieden iterieren können?
            for i in range(prev_i, annotation.kernels[0].offset):
                this = observations[i]["name"] + " @ " + observations[i]["place"]

                if this in n_seen:
                    if n_seen[this] == 1:
                        logger.debug(
                            f"Loop found in {annotation.start.name} {annotation.end.param}: {this} ⟳"
                        )
                    n_seen[this] += 1
                else:
                    n_seen[this] = 1

                if not prev in delta:
                    delta[prev] = set()
                delta[prev].add(this)

                # annotation.start.param may be incomplete, for instance in cases
                # where DPUs are allocated before the input file is loadeed (and
                # thus before the problem size is known).
                # Hence, we must use annotation.end.param whenever we deal
                # with possibly problem size-dependent behaviour.
                if not (prev, this) in delta_param:
                    delta_param[(prev, this)] = set()
                delta_param[(prev, this)].add(
                    utils.param_dict_to_str(annotation.end.param)
                )

                prev = this
                prev_i = i + 1

                total_latency_us += observations[i]["attribute"].get("latency_us", 0)

                meta_observations.append(
                    {
                        "name": f"__trace__ {this}",
                        "param": annotation.end.param,
                        "attribute": dict(
                            filter(
                                lambda kv: not kv[0].startswith("e_"),
                                observations[i]["param"].items(),
                            )
                        ),
                    }
                )
            prev_non_kernel = prev

        for kernel in annotation.kernels:
            prev = prev_non_kernel
            for i in range(prev_i, kernel.offset):
                this = observations[i]["name"] + " @ " + observations[i]["place"]

                if not prev in delta:
                    delta[prev] = set()
                delta[prev].add(this)

                if not (prev, this) in delta_param:
                    delta_param[(prev, this)] = set()
                delta_param[(prev, this)].add(
                    utils.param_dict_to_str(annotation.end.param)
                )

                # The last iteration (next block) contains a single kernel,
                # so we do not increase total_latency_us here.
                # However, this means that we will only ever get one latency
                # value for each set of kernels with a common problem size,
                # despite potentially having far more data at our fingertips.
                # We could provide one total_latency_us for each kernel
                # (by combining start latency + kernel latency + teardown latency),
                # but for that we first need to distinguish between kernel
                # components and teardown components in the following block.

                prev = this
                prev_i = i + 1

                meta_observations.append(
                    {
                        "name": f"__trace__ {this}",
                        "param": annotation.end.param,
                        "attribute": dict(
                            filter(
                                lambda kv: not kv[0].startswith("e_"),
                                observations[i]["param"].items(),
                            )
                        ),
                    }
                )

        # There is no kernel end signal in the underlying data, so the last iteration also contains a kernel run.
        prev = prev_non_kernel
        for i in range(prev_i, annotation.end.offset):
            this = observations[i]["name"] + " @ " + observations[i]["place"]

            if this in n_seen:
                if n_seen[this] == 1:
                    logger.debug(
                        f"Loop found in {annotation.start.name} {annotation.end.param}: {this} ⟳"
                    )
                n_seen[this] += 1
            else:
                n_seen[this] = 1

            if not prev in delta:
                delta[prev] = set()
            delta[prev].add(this)

            if not (prev, this) in delta_param:
                delta_param[(prev, this)] = set()
            delta_param[(prev, this)].add(utils.param_dict_to_str(annotation.end.param))

            total_latency_us += observations[i]["attribute"].get("latency_us", 0)

            prev = this

            meta_observations.append(
                {
                    "name": f"__trace__ {this}",
                    "param": annotation.end.param,
                    "attribute": dict(
                        filter(
                            lambda kv: not kv[0].startswith("e_"),
                            observations[i]["param"].items(),
                        )
                    ),
                }
            )

        if not prev in delta:
            delta[prev] = set()
        delta[prev].add("__end__")
        if not (prev, "__end__") in delta_param:
            delta_param[(prev, "__end__")] = set()
        delta_param[(prev, "__end__")].add(
            utils.param_dict_to_str(annotation.end.param)
        )

        for transition, count in n_seen.items():
            meta_observations.append(
                {
                    "name": f"__loop__ {transition}",
                    "param": annotation.end.param,
                    "attribute": {"n_iterations": count},
                }
            )

        if total_latency_us:
            meta_observations.append(
                {
                    "name": annotation.start.name,
                    "param": annotation.end.param,
                    "attribute": {"latency_us": total_latency_us},
                }
            )

        is_loop = dict(
            map(lambda kv: (kv[0], True), filter(lambda kv: kv[1] > 1, n_seen.items()))
        )

        return delta, delta_param, meta_observations, is_loop


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
                logger.warning(
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
                if use_lut:
                    logger.error(
                        f"Cannot predict {name}.{action}({param}) from LUT model"
                    )
                else:
                    logger.error(f"Cannot predict {name}.{action}({param}) from model")
                raise
            except TypeError:
                if not use_lut:
                    logger.error(f"Cannot predict {name}.{action}({param}) from model")
                raise

            if aggregate == "sum":
                aggregate_init += event_output
            else:
                raise RuntimeError(f"Unknown aggregate type: {aggregate}")

        return aggregate_init
