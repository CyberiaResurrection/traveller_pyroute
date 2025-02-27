# distutils: language = c++
# cython: profile=True
"""
Created on Feb 27, 2025

@author: CyberiaResurrection

Implementation of the NBA* (New Bidirectional A*) algorithm for bidirectional
A*-like pathfinding.  This code:
    Leans on numpy to handle neighbour nodes, edges to same and heuristic values in bulk
    Tracks upper-bounds on shortest-path cost as they are found
    Prunes neighbour candidates early - if this exhausts a node by leaving it no viable neighbour candidates, so be it
    Takes an optional externally-supplied upper bound
        - Sanity and correctness of this upper bound are the _caller_'s responsibility
        - If the supplied upper bound produces a pathfinding failure, so be it
"""
#import cython
#from cython.cimports.numpy import numpy as cnp
#from cython.cimports.minmaxheap import MinMaxHeap, astar_t
from heapq import heappop, heappush, heapify

import networkx as nx
import numpy as np

#cnp.import_array()

float64max = np.finfo(np.float64).max


#@cython.cdivision(True)
def _calc_branching_factor(nodes_queued: int, path_len: int):
    old: float
    new: float
    rhs: float
    power: float
    if path_len == nodes_queued or 1 > path_len or 1 > nodes_queued:
        return 1.0

    power = 1.0 / path_len
    # Letting nodes_queued be S, and path_len be d, we're trying to solve for the value of r in the following:
    # S = r * ( r ^ (d-1) - 1 ) / ( r - 1 )
    # Applying some sixth-grade algebra:
    # Sr - S = r * ( r ^ (d-1) - 1 )
    # Sr - S = r ^ d - r
    # Sr - S + r = r ^ d
    # r ^ d = Sr - S + r
    #
    # That final line is an ideal form to apply fixed-point iteration to, starting with an initial guess for r
    # and feeding it into:
    # r* = (Sr - S + r) ^ (1/d)
    # iterating until r* and r sufficiently converge.

    old = 0.0
    new = 0.5 * (1 + nodes_queued ** power)
    while 0.001 <= abs(new - old):
        old = new
        rhs = nodes_queued * new - nodes_queued + new
        new = rhs ** power

    return round(new, 3)


#@cython.boundscheck(False)
#@cython.initializedcheck(False)
#@cython.wraparound(False)
#@cython.nonecheck(False)
def bidir_path_numpy(G, source: int, target: int, bulk_heuristic,
                     upbound: float = float64max, diagnostics: bool = False):
    G_succ: list[tuple[list[int], list[float]]]
    potential_fwd: list[float]
    potential_rev: list[float]
    # upbound: cython.float
    distances_fwd: np.ndarray[float, 1]
    distances_rev: np.ndarray[float, 1]
    G_succ = G._arcs  # For speed-up

    # pre-calc heuristics for all nodes to the target node
    potential_fwd = bulk_heuristic(target)
    potential_rev = bulk_heuristic(source)

    # Maps explored nodes to parent closest to the source.
    explored_fwd: dict[int, int] = {}
    explored_rev: dict[int, int] = {}

    # Traces lowest distance from source node found for each node
    distances_fwd = np.ones((len(G_succ)), dtype=float) * float64max
    distances_fwd[source] = 0.0
    distances_rev = np.ones((len(G_succ)), dtype=float) * float64max
    distances_rev[target] = 0.0

    # track forward and reverse queues
    queue_fwd = [(potential_fwd[source], 0, source, None)]
    queue_rev = [(potential_rev[target], 0, target, None)]
    f_fwd = potential_fwd[source]
    f_rev = potential_rev[target]
    oldbound = upbound

    while queue_fwd and queue_rev:
        if len(queue_rev) < len(queue_fwd):
            queue_rev, explored_rev, distances_rev, upbound = bidir_iteration(G_succ, diagnostics, queue_rev,
                                                                              explored_rev, distances_rev, distances_fwd,
                                                                              potential_rev, potential_fwd,
                                                                              target, source, upbound, f_fwd)
            if queue_rev:
                f_rev = queue_rev[0][0]
        else:
            queue_fwd, explored_fwd, distances_fwd, upbound = bidir_iteration(G_succ, diagnostics, queue_fwd,
                                                                              explored_fwd, distances_fwd, distances_rev,
                                                                              potential_fwd, potential_rev,
                                                                              source, target, upbound, f_rev)
            if queue_fwd:
                f_fwd = queue_fwd[0][0]

        if oldbound > upbound:
            oldbound = upbound
            queue_fwd = [item for item in queue_fwd if item[0] <= upbound]
            queue_rev = [item for item in queue_rev if item[0] <= upbound]
            heapify(queue_fwd)
            heapify(queue_rev)
        #bestpath, diag = astar_numpy_core(G_succ, diagnostics, distances, potential_fwd, source, target, float64max)

    bestpath = []
    diag = {}
    if 0 == len(bestpath):
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    return bestpath, diag


def bidir_iteration(G_succ: list[tuple[list[int], list[float]]], diagnostics: bool, queue: list[tuple],
                    explored: dict[int, int], distances: np.ndarray[float], distances_other: np.ndarray[float],
                    potentials: np.ndarray[float], potentials_other: np.ndarray[float], source: int, target: int,
                    upbound: float, f_other: float):
    # Pop the smallest item from queue.
    _, dist, curnode, parent = heappop(queue)

    if curnode in explored:
        # Do not override the parent of starting node
        if explored[curnode] == -1:
            return queue, explored, distances, upbound
        # We've found a bad path, just move on
        qcost = distances[curnode]
        if qcost <= dist:
            return queue, explored, distances, upbound
        # If we've found a better path, update
        #revis_continue += 1
        distances[curnode] = dist

    explored[curnode] = parent

    active_nodes = G_succ[curnode][0]
    active_costs = G_succ[curnode][1]

    targdex = -1
    num_nodes = len(active_nodes)
    for i in range(num_nodes):
        act_nod = active_nodes[i]

    # Now unconditionally queue _all_ nodes that are still active, worrying about filtering out the bound-busting
    # neighbours later.
    counter = 0
    for i in range(num_nodes):
        act_nod = active_nodes[i]
        act_wt = dist + active_costs[i]
        if act_wt >= distances[act_nod]:
            continue
        aug_wt = act_wt + potentials[act_nod]
        if aug_wt > upbound:
            continue
        if act_wt + f_other - potentials_other[act_nod] > upbound:
            continue
        distances[act_nod] = act_wt
        heappush(queue, (aug_wt, act_wt, act_nod, curnode))
        upbound = min(upbound, act_wt + distances_other[act_nod])

    return queue, explored, distances, upbound


#@cython.cfunc
#@cython.infer_types(True)
#cython.boundscheck(False)
#@cython.initializedcheck(False)
#@cython.nonecheck(False)
#@cython.wraparound(False)
#@cython.returns(tuple[list[cython.int], dict])
#def astar_numpy_core(G_succ: list[tuple[list[int], list[float]]], diagnostics: bool,
#                     distances: cnp.ndarray[cython.float], potentials: cnp.ndarray[cython.float], source: cython.int,
#                     target: cython.int, upbound: cython.float):
