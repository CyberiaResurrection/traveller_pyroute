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
import cython
from cython.cimports.numpy import numpy as cnp
from cython.cimports.minmaxheap import MinMaxHeap, astar_t
from heapq import heappop, heappush, heapify

import networkx as nx
import numpy as np

cnp.import_array()

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
    potential_fwd: cnp.ndarray[cython.float]
    potential_rev: cnp.ndarray[cython.float]
    # upbound: cython.float
    distances_fwd: np.ndarray[float, 1]
    distances_rev: np.ndarray[float, 1]
    G_succ = G._arcs  # For speed-up

    # pre-calc heuristics for all nodes to the target node
    potential_fwd = bulk_heuristic(target)
    potential_rev = bulk_heuristic(source)
    potential_fwd_view: cython.double[:] = potential_fwd
    potential_rev_view: cython.double[:] = potential_rev

    # Maps explored nodes to parent closest to the source.
    explored_fwd: dict[cython.int, cython.int] = {}
    explored_rev: dict[cython.int, cython.int] = {}

    # Traces lowest distance from source node found for each node
    distances_fwd = np.ones((len(G_succ)), dtype=float) * float64max
    distances_fwd[source] = 0.0
    distances_rev = np.ones((len(G_succ)), dtype=float) * float64max
    distances_rev[target] = 0.0

    # track forward and reverse queues
    queue_fwd = [(potential_fwd[source], 0, source, -1)]
    queue_rev = [(potential_rev[target], 0, target, -1)]
    f_fwd = potential_fwd[source]
    f_rev = potential_rev[target]
    oldbound = upbound

    # track smallest node in both distance arrays
    smalldex: cython.int = -1

    while queue_fwd and queue_rev:
        if len(queue_rev) < len(queue_fwd):
            upbound, mindex = bidir_iteration(G_succ, diagnostics, queue_rev, explored_rev, distances_rev,
                                            distances_fwd, potential_rev, potential_fwd, upbound, f_fwd)
            if queue_rev:
                f_rev = queue_rev[0][0]
            if -1 != mindex:
                smalldex = mindex
        else:
            upbound, mindex = bidir_iteration(G_succ, diagnostics, queue_fwd, explored_fwd, distances_fwd,
                                              distances_rev, potential_fwd, potential_rev, upbound, f_rev)
            if queue_fwd:
                f_fwd = queue_fwd[0][0]
            if -1 != mindex:
                smalldex = mindex

        if oldbound > upbound:
            oldbound = upbound
            queue_fwd = [item for item in queue_fwd if item[0] <= upbound]
            queue_rev = [item for item in queue_rev if item[0] <= upbound]
            heapify(queue_fwd)
            heapify(queue_rev)

    if -1 == smalldex:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    explored_fwd = bidir_fix_explored(explored_fwd, distances_fwd, G_succ, smalldex)
    explored_rev = bidir_fix_explored(explored_rev, distances_rev, G_succ, smalldex)
    bestpath = bidir_build_path(explored_fwd, explored_rev, smalldex)
    diag = {}

    return bestpath, diag


def bidir_iteration(G_succ: list[tuple[list[int], list[float]]], diagnostics: bool, queue: list[tuple],
                    explored: dict[int, int], distances: np.ndarray[float], distances_other: np.ndarray[float],
                    potentials: np.ndarray[float], potentials_other: np.ndarray[float], upbound: float, f_other: float):
    # Pop the smallest item from queue.
    _, dist, curnode, parent = heappop(queue)
    mindex = -1

    if curnode in explored:
        # Do not override the parent of starting node
        if explored[curnode] == -1:
            return upbound, mindex
        # We've found a bad path, just move on
        qcost = distances[curnode]
        if qcost <= dist:
            return upbound, mindex
        # If we've found a better path, update
        #revis_continue += 1
        distances[curnode] = dist

    explored[curnode] = parent

    active_nodes = G_succ[curnode][0]
    active_costs = G_succ[curnode][1]

    num_nodes = len(active_nodes)

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
        rawbound = act_wt + distances_other[act_nod]
        if upbound > rawbound:
            upbound = rawbound
            mindex = act_nod

    return upbound, mindex


def bidir_fix_explored(explored, distances, G_succ, smalldex: cython.int) -> dict:
    if smalldex not in explored:
        active_nodes = G_succ[smalldex][0]
        active_costs = G_succ[smalldex][1]
        skipcost = float64max

        for i in range(len(active_nodes)):
            act_nod = active_nodes[i]
            act_wt = distances[act_nod] + active_costs[i]
            if act_nod in explored and skipcost > act_wt:
                explored[smalldex] = act_nod
                skipcost = act_wt

    return explored


def bidir_build_path(explored_fwd: dict, explored_rev: dict, smalldex: cython.int) -> list[int]:
    path = [smalldex]
    node = explored_fwd[smalldex]
    while node != -1:
        assert node not in path, "Node " + str(node) + " duplicated in discovered path"
        path.append(node)
        node = explored_fwd[node]
    path.reverse()

    node = explored_rev[smalldex]
    while node != -1:
        assert node not in path, "Node " + str(node) + " duplicated in discovered path"
        path.append(node)
        node = explored_rev[node]

    return path


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
