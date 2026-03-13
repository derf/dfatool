#!/usr/bin/env python3

import logging
import numpy as np
from . import utils
from .model import ModelAttribute
from . import functions as df

logger = logging.getLogger(__name__)


class SDKBehaviourModel:

    def __init__(
        self, observations, annotations, unroll_loops=False, show_progress=False
    ):

        self.unroll_loops = unroll_loops
        self.show_progress = show_progress

        meta_observations = list()
        delta_by_name = dict()
        delta_param_by_name = dict()

        if show_progress:
            from progress.bar import Bar

            bar = Bar("Build CFG", max=len(annotations))
            bar.start()

        for annotation in annotations:
            # annotation.start.param may be incomplete, for instance in cases
            # where DPUs are allocated before the input file is loaded (and
            # thus before the problem size is known).
            # However, annotation.end.param may also differ from annotation.start.param (it should not, but that's how some benchmarks roll).
            # So, we use annotation.start.param if it has the same keys as annotation.end.param, and annotation.end.param otherwise
            if sorted(annotation.start.param.keys()) == sorted(
                annotation.end.param.keys()
            ):
                am_tt_param_names = sorted(annotation.start.param.keys())
            else:
                am_tt_param_names = sorted(annotation.end.param.keys())
            am_tt_param_names = sorted(am_tt_param_names + ["#"])
            if annotation.name not in delta_by_name:
                delta_by_name[annotation.name] = dict()
                delta_param_by_name[annotation.name] = dict()
            meta_obs = self.learn_pta(
                observations,
                annotation,
                delta_by_name[annotation.name],
                delta_param_by_name[annotation.name],
            )
            meta_observations += meta_obs
            if self.show_progress:
                bar.next()
        if self.show_progress:
            bar.finish()

        self.am_tt_param_names = am_tt_param_names
        self.delta_by_name = delta_by_name
        self.delta_param_by_name = delta_param_by_name
        self.meta_observations = meta_observations

        self.build_transition_guards()

    def cleanup(self):
        # del self.am_tt_param_names
        # del self.delta_by_name
        # del self.delta_param_by_name
        del self.meta_observations

    def build_transition_guards(self):
        self.transition_guard = dict()

        if self.show_progress:
            from progress.bar import Bar

            bar = Bar(
                "Learn Feature Guards",
                max=sum(map(lambda x: len(x.keys()), self.delta_by_name.values())),
            )
            bar.start()
        for name in sorted(self.delta_by_name.keys()):
            for t_from, t_to_set in self.delta_by_name[name].items():
                i_to_transition = dict()
                delta_param_sets = list()
                to_names = list()
                transition_guard = dict()

                if len(t_to_set) == 1:
                    (t_to,) = list(t_to_set)
                    transition_guard[t_to] = list()
                elif len(t_to_set) > 1:
                    dmt_X = list()
                    dmt_y = list()
                    for i, t_to in enumerate(sorted(t_to_set)):
                        for param_str in self.delta_param_by_name[name][(t_from, t_to)]:
                            dmt_X.append(
                                utils.param_dict_to_list(
                                    utils.param_str_to_dict(param_str),
                                    self.am_tt_param_names,
                                )
                            )
                            dmt_y.append(i)
                            i_to_transition[i] = t_to

                    ma = ModelAttribute(
                        name, t_from, dmt_y, dmt_X, self.am_tt_param_names
                    )
                    ma.build_rmt(
                        with_function_leaves=False,
                        threshold=0,
                        prune=True,
                        prune_scalar=True,
                    )

                    for i, t_to in enumerate(sorted(t_to_set)):
                        for param_str in self.delta_param_by_name[name][(t_from, t_to)]:
                            actual_target = i
                            predicted_target = ma.model_function.eval(
                                utils.param_dict_to_list(
                                    utils.param_str_to_dict(param_str),
                                    self.am_tt_param_names,
                                )
                            )
                            if actual_target != predicted_target:
                                logger.warning(
                                    f"""Model for {name} {t_from}: prediction for {param} is {i_to_transition.get(predicted_target, f"Invalid<{predicted_target}>")}, should be {i_to_transition[actual_target]}"""
                                )

                    if type(ma.model_function) in (
                        df.SplitFunction,
                        df.ScalarSplitFunction,
                    ):
                        flat_model = ma.model_function.flatten()
                    elif type(ma.model_function) is df.StaticFunction:
                        if ma.model_function.value != int(ma.model_function.value):
                            logger.warning(
                                f"Raw feature guard for for {name} {t_from} is {ma.model_function.value}, so either {i_to_transition[int(np.floor(ma.model_function.value))]} or {i_to_transition[int(np.ceil(ma.model_function.value))]}"
                            )
                        transition_name = i_to_transition[int(ma.model_function.value)]
                        transition_guard[transition_name] = list()
                        flat_model = list()
                        logger.warning(
                            f"Model for {name} {t_from} is {ma.model_function}, expected SplitFunction"
                        )
                    else:
                        transition_guard = None
                        flat_model = list()
                        logger.warning(
                            f"Model for {name} {t_from} is {ma.model_function}, expected SplitFunction"
                        )

                    for prefix, output in flat_model:
                        if output != int(output):
                            logger.warning(
                                f"Raw feature guard for for {name} {t_from} {prefix} is {output}, so either {i_to_transition[int(np.floor(output))]} or {i_to_transition[int(np.ceil(output))]}"
                            )
                        transition_name = i_to_transition[int(output)]
                        if transition_name not in transition_guard:
                            transition_guard[transition_name] = list()
                        transition_guard[transition_name].append(prefix)

                self.transition_guard[t_from] = transition_guard
                if self.show_progress:
                    bar.next()
        if self.show_progress:
            bar.finish()

    def get_trace(self, name, in_param_dict):
        delta = self.delta_by_name[name]
        param_dict = in_param_dict.copy()
        param_dict["#"] = 1
        current_state = "__init__"
        trace = [(current_state, param_dict.copy())]
        seen = dict()
        while current_state != "__end__":
            next_states = delta[current_state]

            param_dict["#"] = seen[current_state] = seen.get(current_state, 0) + 1

            if len(next_states) == 0:
                raise RuntimeError(
                    f"get_trace({name}, {in_param_dict}): no outbound edges at {trace}"
                )

            # self.transition_guard[current_state] = φ₁ ∨ φ₂ ∨ … (disjunction)
            # Each condition ∈ self.transition_guard[current_state] = φ₁ ∧ φ₂ ∧ … (conjunction)
            # → as soon as we have found a valid condition, we're good (and can short-circuit)

            if len(next_states) > 1 and self.transition_guard[current_state]:
                matching_next_states = list()
                for candidate in next_states:
                    for condition in self.transition_guard[current_state][candidate]:
                        valid = True
                        for key, equality, value in condition:
                            if equality == "<" and not param_dict[key] < value:
                                valid = False
                                break
                            if equality == "<=" and not param_dict[key] <= value:
                                valid = False
                                break
                            if (
                                equality == "=="
                                and type(value) in (float, int, str)
                                and not param_dict[key] == value
                            ):
                                valid = False
                                break
                            if (
                                equality == "=="
                                and type(value) is tuple
                                and not param_dict[key] in value
                            ):
                                valid = False
                                break
                            if equality == ">=" and not param_dict[key] >= value:
                                valid = False
                                break
                            if equality == ">" and not param_dict[key] > value:
                                valid = False
                                break
                        if valid:
                            matching_next_states.append(candidate)
                            break
                next_states = matching_next_states

            if len(next_states) == 0:
                logger.error(
                    f"get_trace({name}, {param_dict}): found no valid outbound transitions"
                )
                logger.error(f"    trace = {trace}")
                for candidate in self.transition_guard[current_state]:
                    logger.error(f"    candidate {candidate}")
                # raise RuntimeError("found no valid outbound transitions")
                next_states = ["__end__"]
            if len(next_states) > 1:
                logger.error(
                    f"get_trace({name}, {param_dict}): found non-deterministic outbound transitions"
                )
                logger.error(f"    trace = {trace}")
                logger.error(f"    candidates = {next_states}")
                #  raise RuntimeError("found non-deterministic outbound transitions")
                next_states = ["__end__"]

            (next_state,) = next_states

            trace.append((next_state, param_dict.copy()))
            current_state = next_state

        return trace

    def learn_pta(self, observations, annotation, delta=dict(), delta_param=dict()):
        prev_i = annotation.start.offset
        prev = "__init__"
        prev_non_kernel = prev
        meta_observations = list()
        n_seen = {"__init__": 1}

        total_latency_us = 0
        total_latency_ms = 0

        if sorted(annotation.start.param.keys()) == sorted(annotation.end.param.keys()):
            param_dict = annotation.start.param
        else:
            param_dict = annotation.end.param

        param_dict["#"] = 1

        if annotation.kernels:
            # ggf. als dict of tuples, für den Fall dass Schleifen verschieden iterieren können?
            for i in range(prev_i, annotation.kernels[0].offset):
                this = observations[i]["name"] + " @ " + observations[i]["place"]

                if self.unroll_loops:
                    this = this + " #" + str(n_seen.get(this, 0))

                if not prev in delta:
                    delta[prev] = set()
                delta[prev].add(this)

                if not (prev, this) in delta_param:
                    delta_param[(prev, this)] = set()
                param_dict["#"] = n_seen[prev]
                param_str = utils.param_dict_to_str(param_dict)
                delta_param[(prev, this)].add(param_str)

                prev = this
                prev_i = i + 1

                total_latency_us += observations[i]["attribute"].get("latency_us", 0)
                total_latency_ms += observations[i]["attribute"].get("latency_ms", 0)

                # must happen after setting param_dict["#"] in case this == prev
                if this in n_seen:
                    if n_seen[this] == 1:
                        logger.debug(
                            f"Loop found in {annotation.start.name} {param_dict}: {this} ⟳"
                        )
                    n_seen[this] += 1
                else:
                    n_seen[this] = 1

                meta_observations.append(
                    {
                        "name": this,
                        "param": param_dict.copy(),
                        "attribute": observations[i]["param"],
                    }
                )
            prev_non_kernel = prev

        for kernel in annotation.kernels:
            prev = prev_non_kernel
            n_seen_kernel = dict()
            for i in range(prev_i, kernel.offset):
                this = observations[i]["name"] + " @ " + observations[i]["place"]

                if self.unroll_loops:
                    this = this + " #" + str(n_seen_kernel.get(this, 0) + 1)

                if not prev in delta:
                    delta[prev] = set()
                delta[prev].add(this)

                if not (prev, this) in delta_param:
                    delta_param[(prev, this)] = set()
                param_dict["#"] = n_seen_kernel.get(prev, n_seen.get(prev))
                param_str = utils.param_dict_to_str(param_dict)
                delta_param[(prev, this)].add(param_str)

                # must happen after setting param_dict["#"] in case this == prev
                if this in n_seen_kernel:
                    n_seen_kernel[this] += 1
                else:
                    n_seen_kernel[this] = 1

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
                        "name": this,
                        "param": param_dict.copy(),
                        "attribute": observations[i]["param"],
                    }
                )

        # There is no kernel end signal in the underlying data, so the last iteration also contains a kernel run.
        prev = prev_non_kernel
        for i in range(prev_i, annotation.end.offset):
            this = observations[i]["name"] + " @ " + observations[i]["place"]

            if self.unroll_loops:
                this = this + " #" + str(n_seen.get(this, 0) + 1)

            if not prev in delta:
                delta[prev] = set()
            delta[prev].add(this)

            if not (prev, this) in delta_param:
                delta_param[(prev, this)] = set()
            param_dict["#"] = n_seen[prev]
            param_str = utils.param_dict_to_str(param_dict)
            delta_param[(prev, this)].add(param_str)

            total_latency_us += observations[i]["attribute"].get("latency_us", 0)
            total_latency_ms += observations[i]["attribute"].get("latency_ms", 0)

            # must happen after setting param_dict["#"] in case this == prev
            if this in n_seen:
                if n_seen[this] == 1:
                    logger.debug(
                        f"Loop found in {annotation.start.name} {param_dict}: {this} ⟳"
                    )
                n_seen[this] += 1
            else:
                n_seen[this] = 1

            prev = this

            meta_observations.append(
                {
                    "name": this,
                    "param": param_dict.copy(),
                    "attribute": observations[i]["param"],
                }
            )

        if not prev in delta:
            delta[prev] = set()
        delta[prev].add("__end__")
        if not (prev, "__end__") in delta_param:
            delta_param[(prev, "__end__")] = set()
        param_dict["#"] = 1
        param_str = utils.param_dict_to_str(param_dict)
        delta_param[(prev, "__end__")].add(param_str)

        if total_latency_us:
            param_dict.pop("#")
            meta_observations.append(
                {
                    "name": annotation.start.name,
                    "param": param_dict,
                    "attribute": {"latency_us": total_latency_us},
                }
            )
        elif total_latency_ms:
            param_dict.pop("#")
            meta_observations.append(
                {
                    "name": annotation.start.name,
                    "param": param_dict,
                    "attribute": {"latency_ms": total_latency_ms},
                }
            )

        return meta_observations


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
