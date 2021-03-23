#!/usr/bin/env python3

import numpy as np
import os
import scipy
from bisect import bisect_left, bisect_right


def compensate(data, timestamps, event_timestamps, offline_index=None):
    """Use ruptures (e.g. Pelt, Dynp) to determine transition timestamps."""
    from dfatool.pelt import PELT

    # "rbf" und "l2" scheinen ähnlich gut zu funktionieren, l2 ist schneller. l1 ist wohl noch besser.
    # PELT does not find changepoints for transitions which span just four or five data points (i.e., transitions shorter than ~2ms).
    # Workaround: Double the data rate passed to PELT by interpolation ("stretch=2")
    pelt = PELT(with_multiprocessing=False, stretch=2, min_dist=1, cache_dir=None)
    expected_transition_start_timestamps = event_timestamps[::2]
    transition_start_candidate_weights = list()
    drift = 0

    # TODO auch Kandidatenbestimmung per Ableitung probieren
    # (-> Umgebungsvariable zur Auswahl)

    pelt_traces = list()
    range_timestamps = list()
    candidate_weights = list()

    for i, expected_start_ts in enumerate(expected_transition_start_timestamps):
        expected_end_ts = event_timestamps[2 * i + 1]
        # assumption: maximum deviation between expected and actual timestamps is 5ms.
        # We use ±10ms to have some contetx for PELT
        et_timestamps_start = bisect_left(timestamps, expected_start_ts - 10e-3)
        et_timestamps_end = bisect_right(timestamps, expected_end_ts + 10e-3)
        range_timestamps.append(timestamps[et_timestamps_start : et_timestamps_end + 1])
        pelt_traces.append(data[et_timestamps_start : et_timestamps_end + 1])

        # TODO for greedy mode, perform changepoint detection between greedy steps
        # (-> the expected changepoint area is well-known, Dynp with 1/2 changepoints
        # should work much better than "somewhere in these 20ms there should be a transition")

    if os.getenv("DFATOOL_DRIFT_COMPENSATION_PENALTY"):
        penalties = (int(os.getenv("DFATOOL_DRIFT_COMPENSATION_PENALTY")),)
    else:
        penalties = (1, 2, 5, 10, 15, 20)
    for penalty in penalties:
        changepoints_by_transition = pelt.get_changepoints(pelt_traces, penalty=penalty)
        for i in range(len(expected_transition_start_timestamps)):
            candidate_weights.append(dict())
            for changepoint in changepoints_by_transition[i]:
                if changepoint in candidate_weights[i]:
                    candidate_weights[i][changepoint] += 1
                else:
                    candidate_weights[i][changepoint] = 1

    for i, expected_start_ts in enumerate(expected_transition_start_timestamps):

        # TODO ist expected_start_ts wirklich eine gute Referenz? Wenn vor einer Transition ein UART-Dump
        # liegt, dürfte expected_end_ts besser sein, dann muss allerdings bei der compensation wieder auf
        # start_ts zurückgerechnet werden.
        transition_start_candidate_weights.append(
            list(
                map(
                    lambda k: (
                        range_timestamps[i][k] - expected_start_ts,
                        range_timestamps[i][k] - expected_end_ts,
                        candidate_weights[i][k],
                    ),
                    sorted(candidate_weights[i].keys()),
                )
            )
        )

    if os.getenv("DFATOOL_COMPENSATE_DRIFT_GREEDY"):
        return compensate_drift_greedy(
            event_timestamps, transition_start_candidate_weights
        )

    return compensate_drift_graph(
        event_timestamps,
        transition_start_candidate_weights,
        offline_index=offline_index,
    )


def compensate_drift_graph(
    event_timestamps, transition_start_candidate_weights, offline_index=None
):
    # Algorithm: Obtain the shortest path in a layered graph made up from
    # transition candidates. Each node represents a transition candidate timestamp, and each layer represents a transition.
    # Each node in layer i contains a directed edge to each node in layer i+1.
    # The edge weight is the drift delta between the two nodes. So, if,
    # node X (transition i, candidate a) has a drift of 5, and node Y
    # (transition i+1, candidate b) has a drift of -2, the weight is 7.
    # The first and last layer of the graph consists of a single node
    # with a drift of 0, representing the start / end synchronization pulse, respectively.

    prev_nodes = [0]
    prev_drifts = [0]
    node_drifts = [0]
    edge_srcs = list()
    edge_dsts = list()
    csr_weights = list()

    # (transition index) -> [candidate 0/start node, candidate 0/end node, candidate 1/start node, ...]
    nodes_by_transition_index = dict()

    # (node number) -> (transition index, candidate index, is_end)
    # (-> transition_start_candidate_weights[transition index][candidate index][is_end])
    transition_by_node = dict()

    compensated_timestamps = list()

    # default: up to two nodes may be skipped
    max_skip_count = 2

    if os.getenv("DFATOOL_DC_MAX_SKIP"):
        max_skip_count = int(os.getenv("DFATOOL_DC_MAX_SKIP"))

    for transition_index, candidates in enumerate(transition_start_candidate_weights):
        new_nodes = list()
        new_drifts = list()
        i_offset = prev_nodes[-1] + 1
        nodes_by_transition_index[transition_index] = list()
        for new_node_i, (new_drift_start, new_drift_end, _) in enumerate(candidates):
            for is_end, new_drift in enumerate((new_drift_start, new_drift_end)):
                new_node = i_offset + new_node_i * 2 + is_end
                nodes_by_transition_index[transition_index].append(new_node)
                transition_by_node[new_node] = (transition_index, new_node_i, is_end)
                new_nodes.append(new_node)
                new_drifts.append(new_drift)
                node_drifts.append(new_drift)
                for prev_node_i, prev_node in enumerate(prev_nodes):
                    prev_drift = prev_drifts[prev_node_i]

                    edge_srcs.append(prev_node)
                    edge_dsts.append(new_node)

                    delta_drift = np.abs(prev_drift - new_drift)
                    # TODO evaluate "delta_drift ** 2" or similar nonlinear
                    # weights -> further penalize large drift deltas
                    csr_weights.append(delta_drift)

        # a transition's candidate list may be empty
        if len(new_nodes):
            prev_nodes = new_nodes
            prev_drifts = new_drifts

    # add an end node for shortest path search
    # (end node == final sync, so drift == 0)
    new_node = prev_nodes[-1] + 1
    for prev_node_i, prev_node in enumerate(prev_nodes):
        prev_drift = prev_drifts[prev_node_i]
        edge_srcs.append(prev_node)
        edge_dsts.append(new_node)
        csr_weights.append(np.abs(prev_drift))

    # Add "skip" edges spanning from transition i to transition i+n (n > 1).
    # These avoid synchronization errors caused by transitions wich are
    # not found by changepiont detection, as long as they are sufficiently rare.
    for transition_index, candidates in enumerate(transition_start_candidate_weights):
        for skip_count in range(2, max_skip_count + 2):
            if transition_index < skip_count:
                continue
            for from_node in nodes_by_transition_index[transition_index - skip_count]:
                for to_node in nodes_by_transition_index[transition_index]:

                    (from_trans_i, from_candidate_i, from_is_end) = transition_by_node[
                        from_node
                    ]
                    to_trans_i, to_candidate_i, to_is_end = transition_by_node[to_node]

                    assert transition_index - skip_count == from_trans_i
                    assert transition_index == to_trans_i

                    from_drift = transition_start_candidate_weights[from_trans_i][
                        from_candidate_i
                    ][from_is_end]
                    to_drift = transition_start_candidate_weights[to_trans_i][
                        to_candidate_i
                    ][to_is_end]

                    edge_srcs.append(from_node)
                    edge_dsts.append(to_node)
                    csr_weights.append(
                        np.abs(from_drift - to_drift) + (skip_count - 1) * 270e-6
                    )

    sm = scipy.sparse.csr_matrix(
        (csr_weights, (edge_srcs, edge_dsts)), shape=(new_node + 1, new_node + 1)
    )
    dm, predecessors = scipy.sparse.csgraph.shortest_path(
        sm, return_predecessors=True, indices=0
    )

    nodes = list()
    pred = predecessors[-1]
    while pred > 0:
        nodes.append(pred)
        pred = predecessors[pred]

    nodes = list(reversed(nodes))

    # first and last node are not included in "nodes" as they represent
    # the start/stop sync pulse (and not a transition with sync candidates)

    prev_transition = -1
    for i, node in enumerate(nodes):
        transition, _, _ = transition_by_node[node]
        drift = node_drifts[node]

        while transition - prev_transition > 1:
            prev_drift = node_drifts[nodes[i - 1]]
            prev_transition += 1
            expected_start_ts = event_timestamps[prev_transition * 2] + prev_drift
            expected_end_ts = event_timestamps[prev_transition * 2 + 1] + prev_drift
            compensated_timestamps.append(expected_start_ts)
            compensated_timestamps.append(expected_end_ts)

        expected_start_ts = event_timestamps[transition * 2] + drift
        expected_end_ts = event_timestamps[transition * 2 + 1] + drift
        compensated_timestamps.append(expected_start_ts)
        compensated_timestamps.append(expected_end_ts)
        prev_transition = transition

    # handle skips over the last few transitions, if any
    transition = len(transition_start_candidate_weights) - 1
    while transition - prev_transition > 0:
        prev_drift = node_drifts[nodes[-1]]
        prev_transition += 1
        expected_start_ts = event_timestamps[prev_transition * 2] + prev_drift
        expected_end_ts = event_timestamps[prev_transition * 2 + 1] + prev_drift
        compensated_timestamps.append(expected_start_ts)
        compensated_timestamps.append(expected_end_ts)

    if os.getenv("DFATOOL_EXPORT_DRIFT_COMPENSATION"):
        import json
        from dfatool.utils import NpEncoder

        expected_transition_start_timestamps = event_timestamps[::2]
        filename = os.getenv("DFATOOL_EXPORT_DRIFT_COMPENSATION")
        filename = f"{filename}.{offline_index}"

        with open(filename, "w") as f:
            json.dump(
                [
                    expected_transition_start_timestamps,
                    transition_start_candidate_weights,
                ],
                f,
                cls=NpEncoder,
            )

    return compensated_timestamps


def compensate_drift_greedy(event_timestamps, transition_start_candidate_weights):
    drift = 0
    expected_transition_start_timestamps = event_timestamps[::2]
    compensated_timestamps = list()

    for i, expected_start_ts in enumerate(expected_transition_start_timestamps):
        candidates = sorted(
            map(
                lambda x: x[0] + expected_start_ts,
                transition_start_candidate_weights[i],
            )
        )
        expected_start_ts += drift
        expected_end_ts = event_timestamps[2 * i + 1] + drift

        # choose the next candidates around the expected sync point.
        start_right_sync = bisect_left(candidates, expected_start_ts)
        start_left_sync = start_right_sync - 1

        end_right_sync = bisect_left(candidates, expected_end_ts)
        end_left_sync = end_right_sync - 1

        if start_right_sync >= 0:
            start_left_diff = expected_start_ts - candidates[start_left_sync]
        else:
            start_left_diff = np.inf

        if start_right_sync < len(candidates):
            start_right_diff = candidates[start_right_sync] - expected_start_ts
        else:
            start_right_diff = np.inf

        if end_left_sync >= 0:
            end_left_diff = expected_end_ts - candidates[end_left_sync]
        else:
            end_left_diff = np.inf

        if end_right_sync < len(candidates):
            end_right_diff = candidates[end_right_sync] - expected_end_ts
        else:
            end_right_diff = np.inf

        drift_candidates = (
            start_left_diff,
            start_right_diff,
            end_left_diff,
            end_right_diff,
        )
        min_drift_i = np.argmin(drift_candidates)
        min_drift = min(drift_candidates)

        if min_drift < 5e-4:
            if min_drift_i % 2 == 0:
                # left
                compensated_timestamps.append(expected_start_ts - min_drift)
                compensated_timestamps.append(expected_end_ts - min_drift)
                drift -= min_drift
            else:
                # right
                compensated_timestamps.append(expected_start_ts + min_drift)
                compensated_timestamps.append(expected_end_ts + min_drift)
                drift += min_drift

        else:
            compensated_timestamps.append(expected_start_ts)
            compensated_timestamps.append(expected_end_ts)

    if os.getenv("DFATOOL_EXPORT_DRIFT_COMPENSATION"):
        import json
        from dfatool.utils import NpEncoder

        expected_transition_start_timestamps = event_timestamps[::2]

        with open(os.getenv("DFATOOL_EXPORT_DRIFT_COMPENSATION"), "w") as f:
            json.dump(
                [
                    expected_transition_start_timestamps,
                    transition_start_candidate_weights,
                ],
                f,
                cls=NpEncoder,
            )

    return compensated_timestamps


def compensate_etplusplus(
    data, timestamps, event_timestamps, statechange_indexes, offline_index=None
):
    """Use hardware state changes reported by EnergyTrace++ to determine transition timestamps."""
    expected_transition_start_timestamps = event_timestamps[::2]
    compensated_timestamps = list()
    drift = 0
    for i, expected_start_ts in enumerate(expected_transition_start_timestamps):
        expected_end_ts = event_timestamps[i * 2 + 1]
        et_timestamps_start = bisect_left(timestamps, expected_start_ts - 10e-3)
        et_timestamps_end = bisect_right(timestamps, expected_start_ts + 10e-3)

        candidate_indexes = list()
        for index in statechange_indexes:
            if et_timestamps_start <= index <= et_timestamps_end:
                candidate_indexes.append(index)

        if len(candidate_indexes) == 2:
            drift = timestamps[candidate_indexes[0]] - expected_start_ts

        compensated_timestamps.append(expected_start_ts + drift)
        compensated_timestamps.append(expected_end_ts + drift)

    return compensated_timestamps
