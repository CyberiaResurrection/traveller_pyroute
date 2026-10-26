# distutils: language = c++
# distutils: sources = _pairing_heap.cc
# cython: profile=True

"""
Created on Oct 27, 2026

@author: CyberiaResurrection

Compared to the ancestral networkx version of astar_path, this code:
    Does _not_ use a count() object reference to break ties, as nodes are directly-comparable integers
    Leans on numpy to handle neighbour nodes, edges to same and heuristic values in bulk
    Tracks upper-bounds on shortest-path cost as they are found
    Prunes neighbour candidates early - if this exhausts a node by leaving it no viable neighbour candidates, so be it
    Takes an optional externally-supplied upper bound
        - Sanity and correctness of this upper bound are the _caller_'s responsibility
        - If the supplied upper bound produces a pathfinding failure, so be it
"""

import math

import networkx as nx
import numpy as np
cimport numpy as cnp

from pairing_heap cimport PairingHeap, Item

cnp.import_array()

cdef double float64max = np.finfo(np.float64).max
cdef int ROOT_NODE = -1


cdef double _calc_branching_factor(int nodes_queued, int path_len):
    cdef double old
    cdef double new
    cdef double rhs
    cdef double power
    cdef double raw

    if path_len == nodes_queued or path_len < 1 or nodes_queued < 1:
        return 1.0
    if path_len == 1:
        return nodes_queued - 1.0
    if path_len == 2:
        raw = 0.5 * (-1 + math.sqrt(1 + 4 * (nodes_queued - 1)))
        return round(raw, 3)

    power = 1.0 / (path_len + 1)

    old = 0.0
    new = 0.5 * (1 + nodes_queued ** power)
    while 0.001 <= abs(new - old):
        old = new
        rhs = nodes_queued * new - nodes_queued + 1
        new = rhs ** power

    return round(new, 3)


cpdef tuple astar_path_numpy(
    object G,
    int source,
    int target,
    object bulk_heuristic,
    double upbound=float64max,
    bint diagnostics=False
):
    cdef list G_succ
    cdef cnp.ndarray[cnp.float64_t, ndim=1] potentials
    cdef cnp.ndarray[cnp.float64_t, ndim=1] distances
    cdef list bestpath
    cdef dict diag

    G_succ = G._arcs  # For speed-up

    # Pre-calculate heuristics for all nodes to the target node.
    potentials = bulk_heuristic(target)
    if potentials is None:
        raise ValueError("Bulk heuristic function cannot be None")

    # Trace the lowest distance from the source node found for each node.
    distances = np.ones(len(G_succ), dtype=np.float64) * upbound

    bestpath, diag = astar_numpy_core(
        G_succ, diagnostics, distances, potentials, source, target, upbound
    )

    if len(bestpath) == 0:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    return bestpath, diag


cdef tuple astar_numpy_core(
    list G_succ,
    bint diagnostics,
    cnp.ndarray[cnp.float64_t, ndim=1] distances,
    cnp.ndarray[cnp.float64_t, ndim=1] potentials,
    int source,
    int target,
    double upbound
):
    cdef double[:] distances_view = distances
    cdef double[:] potentials_view = potentials
    cdef long[:] active_nodes_view
    cdef double[:] active_costs_view

    cdef int node_counter = 0
    cdef int queue_counter = 0
    cdef int revisited = 0
    cdef int g_exhausted = 0
    cdef int f_exhausted = 0
    cdef int new_upbounds = 0
    cdef int targ_exhausted = 0
    cdef int revis_continue = 0
    cdef int act_nod
    cdef int curnode
    cdef int parent
    cdef int counter
    cdef int num_nodes
    cdef int i
    cdef int node

    cdef double dist
    cdef double act_wt
    cdef double aug_wt
    cdef double qcost

    cdef list path = []
    cdef dict diag = {}
    cdef dict explored = {}
    cdef dict current_nodes = {}
    cdef PairingHeap queue
    cdef Item* item
    cdef Item result

    distances_view[source] = 0.0

    item = queue.Insert({
        'augment': potentials_view[source],
        'dist': 0.0,
        'curnode': source,
        'parent_node': ROOT_NODE
    })
    current_nodes[source] = <size_t> item

    while not queue.IsEmpty():
        result = queue.DeleteMin()
        dist = result.dist
        curnode = result.curnode
        parent = result.parent_node
        del current_nodes[curnode]
        node_counter += 1

        if curnode == target:
            path.append(curnode)
            node = parent
            while node != ROOT_NODE:
                assert node not in path, (
                    "Node " + str(node) + " duplicated in discovered path"
                )
                path.append(node)
                node = explored[node]
            path.reverse()

            if not diagnostics:
                return path, diag

            branch = _calc_branching_factor(queue_counter, len(path) - 1)
            neighbour_bound = node_counter - 1 + revis_continue - revisited
            un_exhausted = (
                neighbour_bound - f_exhausted - g_exhausted - targ_exhausted
            )
            diag = {
                'nodes_expanded': node_counter,
                'nodes_queued': queue_counter,
                'branch_factor': branch,
                'num_jumps': len(path) - 1,
                'nodes_revisited': revisited,
                'neighbour_bound': neighbour_bound,
                'new_upbounds': new_upbounds,
                'g_exhausted': g_exhausted,
                'f_exhausted': f_exhausted,
                'un_exhausted': un_exhausted,
                'targ_exhausted': targ_exhausted
            }
            return path, diag

        if curnode in explored:
            revisited += 1

            # Do not override the parent of the starting node.
            if explored[curnode] == ROOT_NODE:
                continue

            # We've found a bad path, so move on.
            qcost = distances_view[curnode]
            if qcost <= dist:
                continue

            # If we've found a better path, update.
            revis_continue += 1
            distances_view[curnode] = dist

        explored[curnode] = parent

        active_nodes_view = G_succ[curnode][0]
        active_costs_view = G_succ[curnode][1]
        num_nodes = len(active_nodes_view)

        # Queue all active nodes; filter out bound-busting neighbours here.
        counter = 0
        for i in range(num_nodes):
            act_nod = active_nodes_view[i]
            act_wt = dist + active_costs_view[i]

            if act_wt > distances_view[act_nod]:
                continue

            aug_wt = act_wt + potentials_view[act_nod]
            if aug_wt > upbound:
                continue

            distances_view[act_nod] = act_wt
            if act_nod in current_nodes:
                item = <Item*> current_nodes[act_nod]
                queue.DecreaseKey(item, aug_wt, act_wt, curnode)
            else:
                item = queue.Insert({
                    'augment': aug_wt,
                    'dist': act_wt,
                    'curnode': act_nod,
                    'parent_node': curnode
                })
                current_nodes[act_nod] = <size_t> item
                counter += 1

            if target == act_nod:
                upbound = aug_wt
                new_upbounds += 1

        if counter == 0:
            g_exhausted += 1
        else:
            queue_counter += counter

    return path, diag