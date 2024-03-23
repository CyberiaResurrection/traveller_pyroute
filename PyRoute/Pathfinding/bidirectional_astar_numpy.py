"""
Created on Mar 23, 2024

@author: CyberiaResurrection

Shortest paths using a bidirectional variant of the A* pathfinding algorithm, New Bidirectional A*
http://repub.eur.nl/pub/16100/ei2009-10.pdf

"""
from heapq import heappop, heappush, heapify

import networkx as nx
import numpy as np

TREE_ROOT = -1
TREE_NONE = -100


def _calc_branching_factor(nodes_queued, path_len):
    if path_len == nodes_queued:
        return 1.0
    if 1 == path_len:
        return nodes_queued

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

    old = 0
    new = 0.5 * (1 + nodes_queued ** (1 / path_len))
    while 0.001 <= abs(new - old):
        old = new
        rhs = nodes_queued * new - nodes_queued + new
        new = rhs ** (1 / path_len)

    return round(new, 3)


def bidirectional_astar_path_numpy(G, source, target, bulk_heuristic, min_cost=None, upbound=None):

    push = heappush
    pop = heappop

    G_succ = G._arcs  # For speed-up

    # pre-calc heuristics for all nodes to the respective direction's target node
    potentials = [bulk_heuristic(G._nodes, target), bulk_heuristic(G._nodes, source)]

    # The queues store priority, cost to reach, node,  and parent.
    # Uses Python heapq to keep in priority order.
    # The nodes themselves, being integers, are directly comparable.
    queue = [[(potentials[0][source], 0, source, None)], [(potentials[1][target], 0, target, None)]]

    # Maps explored nodes to parent closest to the respective direction's source.
    explored = [{}, {}]
    parents = np.ones((len(G), 2), dtype=int) * TREE_NONE
    parents[source, 0] = TREE_ROOT
    parents[target, 1] = TREE_ROOT

    # Traces lowest distance from respective direction's source node found for each node
    distances = np.ones((len(G), 2)) * float('+inf')
    distances[source, 0] = 0
    distances[target, 1] = 0

    # Tracks shortest _complete_ path found so far
    floatinf = float('inf')
    upbound = floatinf if upbound is None else upbound
    bestpath = None
    diagnostics = True

    # pre-calc the minimum-cost edge on each node
    min_cost = np.zeros(len(G))

    # minimum f values for each queue
    min_f = np.zeros(2)
    min_f[0] = potentials[0][source]
    min_f[1] = potentials[1][target]

    node_counter = 0
    queue_counter = 2  # Source and target nodes are already queued
    revisited = 0
    g_exhausted = 0
    f_exhausted = 0
    new_upbounds = 0
    targ_exhausted = 0
    un_exhausted = 0
    revis_continue = 0

    # Now begin the bidirectional pathfinding
    while queue[0] or queue[1]:  # While at least one queued node remains to process, across either queue
        if 0 == len(queue[0]):  # Forward queue is empty, forcing selection of reverse queue
            direction = 1
        elif 0 == len(queue[1]):  # Reverse queue is empty, forcing selection of forward queue
            direction = 0
        else:
            direction = 1 if len(queue[1]) < len(queue[0]) else 0
        other = 1 - direction

        dir_source = target if 1 == direction else source
        dir_target = source if 1 == direction else target

        # Pop the smallest item from current queue.
        estimate, dist, curnode, parent = heappop(queue[direction])
        node_counter += 1
        min_f[direction] = estimate
        min_f_other = min_f[other]
        active_threshold = upbound - min_f_other

        # if curnode busts upbound, or curnode plus shortest path in the other direction busts upbound, move on
        if estimate > upbound or (dist + min_f_other - potentials[other][curnode] > upbound):
            queue[0] = [item for item in queue[0] if item[1] + min_f[1] - potentials[1][item[2]] <= upbound]
            queue[1] = [item for item in queue[1] if item[1] + min_f[0] - potentials[0][item[2]] <= upbound]
            continue

        if curnode in explored[direction]:
            revisited += 1
            # Do not override the parent of starting node
            if explored[direction][curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost = distances[curnode, direction]
            if qcost <= dist:
                queue[direction] = [item for item in queue[direction] if not (item[1] > distances[item[2], direction])]
                queue[other] = [item for item in queue[other] if not (item[1] > distances[item[2], other])]
                continue
            # If we've found a better path, update
            revis_continue += 1
            parents[curnode, direction] = parent
            distances[curnode, direction] = dist

        explored[direction][curnode] = parent

        # Start checking neighbours
        raw_nodes = G_succ[curnode]
        active_nodes = raw_nodes[0]
        active_weights = dist + raw_nodes[1]
        keep = active_weights <= distances[active_nodes, direction]
        active_nodes = active_nodes[keep]
        if 0 == len(active_nodes):
            g_exhausted += 1
            continue
        active_weights = active_weights[keep]
        keep = active_weights - potentials[other][active_nodes] <= active_threshold
        if not keep.all():
            active_nodes = active_nodes[keep]
            if 0 == len(active_nodes):
                f_exhausted += 1
                continue
            active_weights = active_weights[keep]

        augmented_weights = active_weights + potentials[direction][active_nodes]
        keep = augmented_weights <= upbound
        if not keep.all():
            active_nodes = active_nodes[keep]
            if 0 == len(active_nodes):
                f_exhausted += 1
                continue
            active_weights = active_weights[keep]
            augmented_weights = augmented_weights[keep]

        num_neighbours = len(active_nodes)
        parents[active_nodes, direction] = curnode
        distances[active_nodes, direction] = active_weights

        for i in range(num_neighbours):
            neighbour = active_nodes[i]
            act_weight = active_weights[i]
            aug_weight = augmented_weights[i]
            if aug_weight > upbound:
                continue
            if act_weight - potentials[other][neighbour] > active_threshold:
                continue
            # If the neighbour we're looking at has a label in _both_ searches, then the searches have met,
            # and we can start updating a few things
            if floatinf > distances[neighbour, other]:
                candidate_bound = np.sum(distances[neighbour, :])
                if upbound > candidate_bound:
                    new_upbounds += 1
                    neighparent = parents[neighbour, other]
                    neighparent = None if TREE_ROOT == neighparent else neighparent

                    upbound = candidate_bound
                    has_bound = True
                    path = buildpath(curnode, parent, explored[direction])
                    # kludge to work around short-range double-ups
                    if dir_target != path[-1]:
                        revpath = buildpath(neighbour, neighparent, explored[other], False)
                        for item in revpath:
                            path.append(item)
                    bestpath = path
                    queue[0] = [item for item in queue[0] if item[1] + min_f[1] - potentials[1][item[2]] <= upbound]
                    queue[1] = [item for item in queue[1] if item[1] + min_f[0] - potentials[0][item[2]] <= upbound]
            push(queue[direction], (aug_weight, act_weight, neighbour, curnode))
            queue_counter += 1

    if bestpath is None:
        raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    if target == bestpath[0]:
        bestpath.reverse()
    if not diagnostics:
        return bestpath, {}
    branch = _calc_branching_factor(queue_counter, len(bestpath) - 1)
    neighbour_bound = node_counter - 1 + revis_continue - revisited
    un_exhausted = neighbour_bound - f_exhausted - g_exhausted - targ_exhausted
    diagnostics = {'nodes_expanded': node_counter, 'nodes_queued': queue_counter, 'branch_factor': branch,
                   'num_jumps': len(bestpath) - 1, 'nodes_revisited': revisited, 'neighbour_bound': neighbour_bound,
                   'new_upbounds': new_upbounds, 'g_exhausted': g_exhausted, 'f_exhausted': f_exhausted,
                   'un_exhausted': un_exhausted, 'targ_exhausted': targ_exhausted}
    return bestpath, diagnostics


def buildpath(node, parent, explored, reverse=True):
    path = [node]
    node = parent

    while node is not None:
        path.append(node)
        node = explored[node]
    if reverse:
        path.reverse()
    return path
