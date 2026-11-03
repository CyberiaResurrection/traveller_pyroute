# distutils: language = c++
# cython: profile=True
"""
Created on Aug 19, 2024

@author: CyberiaResurrection
"""
import bisect

import cython
from cython.cimports.numpy import numpy as cnp
from cython.cimports.minmaxheap import MinMaxHeap, dijkstra_t

import numpy as np

cnp.import_array()
ROOT_NODE: cython.const[cython.int] = -1


@cython.ccall
@cython.boundscheck(False)
@cython.initializedcheck(True)
@cython.nonecheck(False)
@cython.wraparound(False)
def dijkstra_core(arcs: cython.list[cython.ctuple[cnp.ndarray[cython.int], cnp.ndarray[cython.float]]],
                  distance_labels: cnp.ndarray[cython.float], divisor: cython.float,
                  seeds: cython.list[cython.int],
                  max_neighbour_labels: cnp.ndarray[cython.float], min_cost: cnp.ndarray[cython.float]) -> tuple:
    if not isinstance(min_cost, cnp.ndarray):
        raise ValueError("min_cost must be ndarray")
    if not isinstance(max_neighbour_labels, cnp.ndarray):
        raise ValueError("max_neighbour_labels must be ndarray")
    if not isinstance(distance_labels, cnp.ndarray):
        raise ValueError("distance_labels must be ndarray")
    if not 0 < divisor <= 1.0:
        raise ValueError("divisor must be positive and <= 1.0")

    neighbours: cython.ctuple[cnp.ndarray[cython.int], cnp.ndarray[cython.float]]
    act_wt: cython.float
    act_nod: cython.int
    num_nodes: cython.size_t
    index: cython.size_t
    distance_labels_view: cython.double[:] = distance_labels
    max_neighbour_labels_view: cython.double[:] = max_neighbour_labels
    min_cost_view: cython.double[:] = min_cost
    # Using -100 to track "not considered during processing"
    parents: cnp.ndarray[cython.int] = np.ones(len(arcs), dtype=np.int64) * -100
    parents_view: cython.long[:] = parents
    active_nodes_view: cython.long[:]
    active_costs_view: cython.double[:]
    tail: cython.int
    dist_tail: cython.float
    heap: MinMaxHeap[dijkstra_t]
    nodes_processed: cython.int = 0
    nodes_queued: cython.int = 0
    nodes_exceeded: cython.int = 0
    nodes_min_exceeded: cython.int = 0
    nodes_tailed: cython.int = 0

    heap = MinMaxHeap[dijkstra_t]()
    heap.reserve(len(seeds))
    for index in range(0, len(seeds)):
        act_nod = seeds[index]
        if ROOT_NODE == parents_view[act_nod]:
            continue
        parents_view[act_nod] = ROOT_NODE
        heap.insert({'act_wt': distance_labels_view[act_nod], 'act_nod': act_nod})
        nodes_queued += 1

    while 0 < heap.size():
        result = heap.popmin()
        dist_tail = result.act_wt
        tail = result.act_nod

        if dist_tail > distance_labels_view[tail] or dist_tail + min_cost_view[tail] > max_neighbour_labels_view[tail]:
            if dist_tail > distance_labels_view[tail] - 1e-8:
                nodes_exceeded += 1
            else:
                nodes_min_exceeded += 1
            continue

        nodes_processed += 1

        # Link weights are strictly positive, thus lower bounded by zero. Thus, when the current dist_tail value exceeds
        # the corresponding node's distance label at the other end of the candidate edge, trim that edge.  Such edges
        # cannot _possibly_ result in smaller distance labels.  By a similar argument, filter the remaining edges
        # when the sum of dist_tail and that edge's weight equals or exceeds the corresponding node's distance label.
        neighbours = arcs[tail]
        active_nodes_view = neighbours[0]
        active_costs_view = neighbours[1]

        # Since max_neighbour_label is an upper bound on the neighbour labels, the max label - dist tail is thus
        # an upper bound on the weights that have a _prayer_ of not busting their respective distance labels.
        # If max label - dist_tail would be inserted at index k, then existing costs at index k and above _WILL_
        # bust their upper bound and don't have to be looked at in detail.
        num_nodes = bisect.bisect_left(active_costs_view, max_neighbour_labels_view[tail] - dist_tail)

        # process neighbours first, then update max_label at the end
        for index in range(0, num_nodes):
            act_nod = active_nodes_view[index]
            act_cost = active_costs_view[index]
            if dist_tail + act_cost >= distance_labels_view[act_nod]:
                nodes_tailed += 1
                continue
            act_wt = dist_tail + divisor * act_cost

            distance_labels_view[act_nod] = act_wt
            parents_view[act_nod] = tail
            heap.insert({'act_wt': act_wt, 'act_nod': act_nod})
            nodes_queued += 1

        max_neighbour_labels_view[tail] = max(distance_labels[active_nodes_view])

    diagnostics = {'nodes_processed': nodes_processed, 'nodes_queued': nodes_queued, 'nodes_exceeded': nodes_exceeded,
                   'nodes_min_exceeded': nodes_min_exceeded, 'nodes_tailed': nodes_tailed}
    return distance_labels, parents, max_neighbour_labels, diagnostics
