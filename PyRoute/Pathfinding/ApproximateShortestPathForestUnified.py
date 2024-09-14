"""
Created on Feb 29, 2024

@author: CyberiaResurrection
"""
import cython
from cython.cimports.numpy import numpy as cnp
cnp.import_array()
import copy
import functools

import numpy as np
from PyRoute.Star import Star
from PyRoute.Pathfinding.DistanceGraph import DistanceGraph
from PyRoute.Pathfinding.single_source_dijkstra import implicit_shortest_path_dijkstra_distance_graph


@cython.cclass
class ApproximateShortestPathForestUnified:

    _graph: object
    _source: object
    _epsilon: cython.float
    _divisor: cython.float
    _sources: list
    _seeds: list
    _num_trees: cython.int
    _graph_len: cython.int
    _distances: cnp.ndarray(cython.float, ndim=2)
    _max_labels: cnp.ndarray(cython.float, ndim=2)

    def __init__(self, source, graph, epsilon, sources=None):
        seeds, source, num_trees = self._get_sources(graph, source, sources)
        self._graph = DistanceGraph(graph)
        self._source = source
        self._epsilon = epsilon
        # memoising this because its value gets used _heavily_ in lower bound calcs, called during heuristic generation
        self._divisor = 1 / (1 + epsilon)
        self._sources = sources
        self._seeds = seeds
        self._num_trees = num_trees
        self._graph_len = len(self._graph)
        self._distances = np.ones((self._graph_len, self._num_trees)) * float('+inf')
        self._max_labels = np.ones((self._graph_len, self._num_trees)) * float('+inf')

        min_cost = self._graph.min_cost(list(range(self._graph_len)), 0)
        # spin up initial distances
        for i in range(self._num_trees):
            raw_seeds = self._seeds[i] if isinstance(self._seeds[i], list) else list(self._seeds[i].values())
            self._distances[raw_seeds, i] = 0
            result = implicit_shortest_path_dijkstra_distance_graph(self._graph, self._source,
                                                                                   self._distances[:, i],
                                                                                   seeds=raw_seeds,
                                                                                   min_cost=min_cost,
                                                                                   divisor=self._divisor)
            self._distances[:, i], self._max_labels[:, i] = result

    def lower_bound(self, source, target):
        raw = np.abs(self._distances[source, :] - self._distances[target, :])
        raw = raw[~np.isinf(raw)]
        if 0 == len(raw):
            return 0
        return np.max(raw)

    @cython.infer_types(True)
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def lower_bound_bulk(self, active_nodes, target_node):
        overdrive, fastpath = self._mona_lisa_overdrive(target_node)

        if fastpath:  # Fastpath - all overdrive elements are _finite_, so all rows are retrieved
            # If all nodes are active, we're on the speedypath - no need to slice self._distances
            # and incur 9x relative overhead
            if len(active_nodes) == self._graph_len:
                raw = np.abs(self._distances - self._distances[target_node, :])
            else:
                raw = np.abs(self._distances[active_nodes, :] - self._distances[target_node, :])
        else:
            # if we haven't got _any_ active lines, throw hands up and spit back zeros
            if not overdrive.any():
                return np.zeros(len(active_nodes), dtype=float)
            actives = self._distances[:, overdrive]
            if len(active_nodes) != self._graph_len:
                actives = actives[active_nodes, :]
            target = self._distances[target_node, overdrive]

            raw = np.abs(actives - target)

        return np.max(raw, axis=1)

    def triangle_upbound(self, source, target):
        stardex = source.index if not isinstance(source, int) else source
        targdex = target.index if not isinstance(target, int) else target
        raw = self._distances[stardex, :] + self._distances[targdex, :]
        raw = raw[raw != float('+inf')]

        return np.min(raw) * (1 + self._epsilon)

    #  Gratuitous William Gibson reference is gratuitous.
    @functools.cache
    def _mona_lisa_overdrive(self, target_node):
        result = copy.deepcopy(self._distances[target_node, :] != float('+inf'))
        return result, result.all()

    def update_edges(self, edges):
        dropnodes = set()
        dropspecific = []
        tree_dex = np.array(list(range(self._num_trees)), dtype=int)
        for _ in tree_dex:
            dropspecific.append(set())
        for item in edges:
            left = item[0]
            right = item[1]
            leftdist = self._distances[left, :]
            rightdist = self._distances[right, :]
            rightdist[np.isinf(rightdist)] = 0
            shelf = self._graph._arcs[left]
            keep = shelf[0] == right
            weight = shelf[1][keep][0]
            delta = abs(leftdist - rightdist)
            # Given distance labels, L, on nodes u and v, assuming u's label being smaller,
            # and edge cost between u and v of d(u, v):
            # L(u) + d(u, v) <= L(v)
            # Or, after rearranging,
            # d(u, v) <= L(v) - L(u)
            # u and v do _NOT_ have to be shortest-path parent/child for this bound to hold

            # Allowing an approximation tolerance of epsilon (say 0.1), that bound becomes
            # d(u, v) * (1 + epsilon) <= L(v) - L(u)

            # If that bound no longer holds, it's due to the edge (u, v) having its weight decreased during pathfinding.
            # Tag each incident node as needing updates.

            if np.max(delta) >= weight:
                dropnodes.add(left)
                dropnodes.add(right)
                overdrive = tree_dex[delta >= weight]
                for k in overdrive:
                    dropspecific[k].add(left)
                    dropspecific[k].add(right)

        # if no nodes are to be dropped, nothing to do - bail out
        if 0 == len(dropnodes):
            return

        # Now we're updating at least one tree, grab the current min-cost vector to feed into implicit-dijkstra
        min_cost = self._graph.min_cost(list(range(self._graph_len)), 0)

        # Now we have the nodes incident to edges that bust the (1+eps) approximation bound, feed them into restarted
        # dijkstra to update the approx-SP tree/forest.  Some nodes in dropnodes may well be SP descendants of others,
        # but it wasn't worth the time or complexity cost to filter them out here.
        for i in range(self._num_trees):
            if 0 == len(dropspecific[i]):
                continue
            self._distances[:, i], self._max_labels[:, i] = implicit_shortest_path_dijkstra_distance_graph(self._graph,
                                                                                   self._source,
                                                                                   distance_labels=self._distances[:, i],
                                                                                   seeds=dropspecific[i],
                                                                                   divisor=self._divisor,
                                                                                   min_cost=min_cost,
                                                                                    max_labels=self._max_labels[:, i])

    def expand_forest(self, nu_seeds):
        raw_seeds = nu_seeds if isinstance(nu_seeds, list) else list(nu_seeds.values())
        nu_distances = np.ones((self._graph_len)) * float('+inf')
        nu_distances[raw_seeds] = 0
        nu_distances, nu_max_labels = implicit_shortest_path_dijkstra_distance_graph(self._graph, self._source,
                                                                nu_distances,
                                                                seeds=raw_seeds,
                                                                divisor=self._divisor)
        result = np.zeros((self._graph_len, 1), dtype=float)
        result[:, 0] = list(nu_distances)
        maxresult = np.zeros((self._graph_len, 1), dtype=float)
        maxresult[:, 0] = list(nu_max_labels)
        self._distances = np.append(self._distances, result, 1)
        self._max_labels = np.append(self._distances, maxresult, 1)
        self._num_trees += 1

    def _get_sources(self, graph, source, sources):
        seeds = None
        num_trees = 1
        if isinstance(source, Star) and source.component is None:
            raise ValueError(
                "Source node " + str(source) + " has undefined component.  Has calculate_components() been run?")
        if isinstance(source, int):
            if 'star' not in graph.nodes[source]:
                raise ValueError("Source node # " + str(source) + " does not have star attribute")
            if graph.nodes[source]['star'].component is None:
                raise ValueError(
                    "Source node " + str(graph.nodes[source][
                                             'star']) + " has undefined component.  Has calculate_components() been run?")
            seeds = [[source]]
        if sources is not None:
            for source in sources:
                if isinstance(source, Star) and sources[source].component is None:
                    raise ValueError("Source node " + str(
                        sources[source]) + " has undefined component.  Has calculate_components() been run?")
                if isinstance(source, int):
                    if 'star' not in graph.nodes[source]:
                        raise ValueError("Source node # " + str(source) + " does not have star attribute")
                    if graph.nodes[source]['star'].component is None:
                        raise ValueError(
                            "Source node " + str(graph.nodes[source][
                                                     'star']) + " has undefined component.  Has calculate_components() been run?")
            if isinstance(sources, dict):
                seeds = [list(sources.values())]
            else:
                seeds = sources
                num_trees = len(sources)
        return seeds, source, num_trees

    def lighten_edge(self, u, v, weight):
        self._graph.lighten_edge(u, v, weight)

    @property
    def num_trees(self):
        return self._num_trees
