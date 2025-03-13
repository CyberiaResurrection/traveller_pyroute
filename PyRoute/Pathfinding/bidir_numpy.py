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
from cython.cimports.unordered_map import unordered_map as umap

import networkx as nx
import numpy as np

cnp.import_array()

float64max = np.finfo(np.float64).max


@cython.cdivision(True)
def _calc_branching_factor(nodes_queued: cython.int, path_len: cython.int):
    old: cython.float
    new: cython.float
    rhs: cython.float
    power: cython.float
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


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def bidir_path_numpy(G, source: cython.int, target: cython.int, bulk_heuristic,
                     upbound: cython.float = float64max, diagnostics: cython.bint = False):
    G_succ: list[tuple[cnp.ndarray[cython.int], cnp.ndarray[cython.float]]] = G._arcs  # For speed-up
    potential_fwd: cnp.ndarray[cython.float]
    potential_rev: cnp.ndarray[cython.float]
    distances_fwd: cnp.ndarray[cython.float]
    distances_rev: cnp.ndarray[cython.float]
    i: cython.int
    curnode: cython.int
    mindex: cython.int
    act_nod: cython.int
    act_wt: cython.double  # This as cython.float caused duplicate nodes in paths during testing
    aug_wt: cython.float
    qcost: cython.float
    rawbound: cython.float

    active_nodes: cnp.ndarray[cython.int]
    active_costs: cnp.ndarray[cython.float]

    # pre-calc heuristics for all nodes to the target node
    potential_fwd = bulk_heuristic(target)
    potential_rev = bulk_heuristic(source)
    potential_fwd_view: cython.double[:] = potential_fwd
    potential_rev_view: cython.double[:] = potential_rev

    # Maps explored nodes to parent closest to the source.
    explored_fwd: umap[cython.int, cython.int] = umap[cython.int, cython.int]()
    explored_fwd.reserve(512)
    explored_rev: umap[cython.int, cython.int] = umap[cython.int, cython.int]()
    explored_rev.reserve(512)

    # Traces lowest distance from source node found for each node
    num_nodes: cython.int = len(G_succ)
    distances_fwd = np.ones(num_nodes, dtype=float) * float64max
    distances_fwd_view: cython.double[:] = distances_fwd
    distances_fwd_view[source] = 0.0
    distances_rev = np.ones(num_nodes, dtype=float) * float64max
    distances_rev_view: cython.double[:] = distances_rev
    distances_rev_view[target] = 0.0

    # track forward and reverse queues
    f_fwd: cython.float = potential_fwd_view[source]
    f_rev: cython.float = potential_rev_view[target]
    queue_fwd: MinMaxHeap[astar_t] = MinMaxHeap[astar_t]()
    queue_fwd.reserve(500)
    queue_fwd.insert({'augment': potential_fwd_view[source], 'dist': 0.0, 'curnode': source, 'parent': -1})
    queue_rev: MinMaxHeap[astar_t] = MinMaxHeap[astar_t]()
    queue_rev.reserve(500)
    queue_rev.insert({'augment': potential_rev_view[target], 'dist': 0.0, 'curnode': target, 'parent': -1})
    oldbound = upbound

    # track smallest node in both distance arrays
    smalldex: cython.int = -1

    while queue_fwd.size() > 0 and queue_rev.size() > 0:
        if queue_rev.size() < queue_fwd.size():
            result = queue_rev.popmin()
            dist = result.dist
            curnode = result.curnode
            parent = result.parent
            mindex = -1

            if 0 != explored_rev.count(curnode):
                # Do not override the parent of starting node
                if explored_rev[curnode] == -1:
                    continue
                # We've found a bad path, just move on
                qcost = distances_rev_view[curnode]
                if qcost <= dist:
                    continue
                # If we've found a better path, update
                # revis_continue += 1
                distances_rev_view[curnode] = dist

            explored_rev[curnode] = parent

            active_nodes = G_succ[curnode][0]
            active_nodes_view: cython.long[:] = active_nodes
            active_costs = G_succ[curnode][1]
            active_costs_view: cython.double[:] = active_costs

            num_nodes = len(active_nodes_view)

            for i in range(num_nodes):
                act_nod = active_nodes_view[i]
                act_wt = dist + active_costs_view[i]
                if act_wt >= distances_rev_view[act_nod]:
                    continue
                aug_wt = act_wt + potential_rev_view[act_nod]
                if aug_wt > upbound:
                    continue
                if act_wt + f_fwd - potential_fwd_view[act_nod] > upbound:
                    continue
                distances_rev_view[act_nod] = act_wt
                queue_rev.insert({'augment': aug_wt, 'dist': act_wt, 'curnode': act_nod, 'parent': curnode})
                rawbound = act_wt + distances_fwd_view[act_nod]
                if upbound > rawbound:
                    upbound = rawbound
                    mindex = act_nod

            if queue_rev.size() > 0:
                result = queue_rev.peekmin()
                f_rev = result.augment
            if -1 != mindex:
                smalldex = mindex
        else:
            result = queue_fwd.popmin()
            dist = result.dist
            curnode = result.curnode
            parent = result.parent
            mindex = -1

            if 0 != explored_fwd.count(curnode):
                # Do not override the parent of starting node
                if explored_fwd[curnode] == -1:
                    continue
                # We've found a bad path, just move on
                qcost = distances_fwd_view[curnode]
                if qcost <= dist:
                    continue
                # If we've found a better path, update
                # revis_continue += 1
                distances_fwd_view[curnode] = dist

            explored_fwd[curnode] = parent

            active_nodes = G_succ[curnode][0]
            active_nodes_view: cython.long[:] = active_nodes
            active_costs = G_succ[curnode][1]
            active_costs_view: cython.double[:] = active_costs

            num_nodes = len(active_nodes_view)

            for i in range(num_nodes):
                act_nod = active_nodes_view[i]
                act_wt = dist + active_costs_view[i]
                if act_wt >= distances_fwd_view[act_nod]:
                    continue
                aug_wt = act_wt + potential_fwd_view[act_nod]
                if aug_wt > upbound:
                    continue
                if act_wt + f_rev - potential_rev_view[act_nod] > upbound:
                    continue
                distances_fwd_view[act_nod] = act_wt
                queue_fwd.insert({'augment': aug_wt, 'dist': act_wt, 'curnode': act_nod, 'parent': curnode})
                rawbound = act_wt + distances_rev_view[act_nod]
                if upbound > rawbound:
                    upbound = rawbound
                    mindex = act_nod

            if queue_fwd.size() > 0:
                result = queue_fwd.peekmin()
                f_fwd = result.augment
            if -1 != mindex:
                smalldex = mindex

    if -1 == smalldex:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    explored_fwd[source] = -1
    explored_rev[target] = -1
    active_nodes = G_succ[smalldex][0]
    active_costs = G_succ[smalldex][1]
    active_nodes_view: cython.long[:] = active_nodes
    active_costs_view: cython.double[:] = active_costs

    explored_fwd = bidir_fix_explored(explored_fwd, distances_fwd_view, active_nodes_view, active_costs_view, smalldex)
    explored_rev = bidir_fix_explored(explored_rev, distances_rev_view, active_nodes_view, active_costs_view, smalldex)
    bestpath = bidir_build_path(explored_fwd, explored_rev, smalldex)
    diag = {}

    return bestpath, diag

@cython.cfunc
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def bidir_fix_explored(explored: umap[cython.int, cython.int], distances: cython.double[:],
                       active_nodes: cython.long[:], active_costs: cython.double[:],
                       smalldex: cython.int) -> umap[cython.int, cython.int]:
    if 0 == explored.count(smalldex):
        act_nod: cython.int
        act_wt: cython.float
        skipcost: cython.float = float64max
        mindex: cython.int = -1
        i: cython.int
        num_nodes: cython.int = len(active_nodes)

        for i in range(num_nodes):
            act_nod = active_nodes[i]
            act_wt = distances[act_nod] + active_costs[i]
            if 0 != explored.count(act_nod) and skipcost > act_wt:
                mindex = act_nod
                skipcost = act_wt

        if -1 != mindex:
            assert smalldex != mindex, "Node " + str(mindex) + " will be ancestor of self"
            explored[smalldex] = mindex

    assert 0 != explored.count(smalldex), "Pivot node " + str(smalldex) + " not added to explored dict " + str(explored)

    return explored

@cython.cfunc
@cython.infer_types(True)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def bidir_build_path(explored_fwd: umap[cython.int, cython.int], explored_rev: umap[cython.int, cython.int], smalldex: cython.int) -> list[cython.int]:
    path = [smalldex]
    node = explored_fwd[smalldex]
    while node != -1:
        assert node not in path, "Node " + str(node) + " duplicated in discovered path, " + str(path) + ". Explored: " + str(explored_fwd)
        path.append(node)
        node = explored_fwd[node]
        assert node != explored_fwd[node], "Node " + str(node) + " is own forward ancestor. Explored: " + str(explored_fwd)
    path.reverse()

    node = explored_rev[smalldex]
    while node != -1:
        assert node not in path, "Node " + str(node) + " duplicated in discovered path, " + str(path) + ". Explored: " + str(explored_rev)
        path.append(node)
        node = explored_rev[node]
        assert node != explored_rev[node], "Node " + str(node) + " is own reverse ancestor. Explored: " + str(explored_rev)
    return path
