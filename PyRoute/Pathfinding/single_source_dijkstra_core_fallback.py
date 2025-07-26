"""
Created on 15 Sep, 2024

@author: CyberiaResurrection
"""
import heapq
import numpy as np


def dijkstra_core(arcs, distance_labels, divisor, seeds, max_neighbour_labels, min_cost) -> tuple:
    if not isinstance(min_cost, np.ndarray):
        raise ValueError("min_cost must be ndarray")
    if not isinstance(max_neighbour_labels, np.ndarray):
        raise ValueError("max_neighbour_labels must be ndarray")
    if not isinstance(distance_labels, np.ndarray):
        raise ValueError("distance_labels must be ndarray")

    heap = [(distance_labels[seed], seed) for seed in seeds if 0 < len(arcs[seed][0])]
    heapq.heapify(heap)

    parents = np.ones(len(arcs), dtype=int) * -100  # Using -100 to track "not considered during processing"
    parents[list(seeds)] = -1  # Using -1 to flag "root node of tree"

    while heap:
        dist_tail, tail = heapq.heappop(heap)

        if dist_tail > distance_labels[tail] or dist_tail + min_cost[tail] > max_neighbour_labels[tail]:
            # Since we've just dequeued a bad node (distance exceeding its current label, or too close to max-label),
            # remove other bad nodes from the list to avoid tripping over them later, and chuck out nodes who
            # can't give better distance labels
            if heap:
                heap = [(distance, tail) for (distance, tail) in heap if distance <= distance_labels[tail]
                        and distance + min_cost[tail] <= max_neighbour_labels[tail]]
                heapq.heapify(heap)
            continue

        # Link weights are strictly positive, thus lower bounded by zero. Thus, when the current dist_tail value exceeds
        # the corresponding node's distance label at the other end of the candidate edge, trim that edge.  Such edges
        # cannot _possibly_ result in smaller distance labels.  By a similar argument, filter the remaining edges
        # when the sum of dist_tail and that edge's weight equals or exceeds the corresponding node's distance label.
        neighbours = arcs[tail]
        active_nodes = neighbours[0]
        active_costs = neighbours[1]
        active_labels = distance_labels[active_nodes]
        # It's not worth (time wise) being cute and trying to break this up, forcing jumps in and out of numpy
        keep = active_costs < (active_labels - dist_tail)
        active_nodes = active_nodes[keep]
        num_nodes = len(active_nodes)

        if 0 == num_nodes:
            continue
        active_weights = dist_tail + divisor * active_costs[keep]
        distance_labels[active_nodes] = active_weights

        parents[active_nodes] = tail

        # update max label _after_ neighbours are processed, to minimise the max_label as far as possible
        max_neighbour_labels[tail] = max(distance_labels[neighbours[0]])

        if 1 == num_nodes:
            heapq.heappush(heap, (active_weights[0], active_nodes[0]))
        elif 2 == num_nodes:
            heapq.heappush(heap, (active_weights[0], active_nodes[0]))
            heapq.heappush(heap, (active_weights[1], active_nodes[1]))
        elif 3 == num_nodes:
            heapq.heappush(heap, (active_weights[0], active_nodes[0]))
            heapq.heappush(heap, (active_weights[1], active_nodes[1]))
            heapq.heappush(heap, (active_weights[2], active_nodes[2]))
        else:  # Only cop the iterator overhead if there's at least 4 neighbours to queue
            for index in range(0, num_nodes):
                heapq.heappush(heap, (active_weights[index], active_nodes[index]))

    return distance_labels, parents, max_neighbour_labels
