"""
Created on Mar 19, 2024

@author: CyberiaResurrection
"""
import numpy as np

from PyRoute.DeltaDebug.DeltaDictionary import SectorDictionary, DeltaDictionary
from PyRoute.DeltaDebug.DeltaGalaxy import DeltaGalaxy
from PyRoute.Inputs.ParseStarInput import ParseStarInput
from PyRoute.Pathfinding.DistanceGraph import DistanceGraph
from Tests.baseTest import baseTest


class testDistanceGraph(baseTest):

    def setUp(self) -> None:
        ParseStarInput.deep_space = {}

    def test_min_cost_in_single_component_1(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        galaxy, graph, _, _ = self._setup_graph(sourcefile)
        components = galaxy.trade.components
        self.assertEqual(1, len(components))

        distgraph = DistanceGraph(graph)
        expected = [0.0, 46.0, 24.0, 24.0, 26.0, 26.0, 26.0, 26.0, 26.0, 25.0, 26.0, 27.0, 28.0, 28.0, 27.0, 25.0, 53.0,
                    24.0, 24.0, 82.0, 81.0, 54.0, 25.0, 47.0, 45.0, 79.0, 25.0, 28.0, 25.0, 79.0, 101.0, 28.0, 28.0,
                    25.0, 25.0, 49.0, 45.0]
        actual = distgraph.min_cost(0)
        self.assertEqual(expected, list(actual), 'Unexpected min-cost vector')

        expected_extended = [0.0, 70.0, 48.0, 48.0, 50.0, 26.0, 51.0, 50.0, 26.0, 50.0, 50.0, 51.0, 52.0, 52.0, 52.0,
                             49.0, 77.0, 48.0, 48.0, 108.0, 106.0, 79.0, 49.0, 75.0, 69.0, 103.0, 49.0, 56.0, 50.0, 104.0,
                             148.0, 53.0, 53.0, 50.0, 50.0, 74.0, 70.0]
        extended = distgraph.min_cost(0, indirect=True)
        self.assertEqual(expected_extended, list(extended), 'Unexpected indirect min-cost vector')

        distgraph.lighten_edge(1, 11, 40)
        expected_extended = [0.0, 64.0, 48.0, 48.0, 50.0, 26.0, 51.0, 50.0, 26.0, 50.0, 50.0, 51.0, 52.0, 52.0, 52.0,
                             49.0, 77.0, 48.0, 48.0, 108.0, 106.0, 79.0, 49.0, 75.0, 69.0, 103.0, 49.0, 56.0, 50.0, 104.0,
                             148.0, 53.0, 53.0, 50.0, 50.0, 74.0, 70.0]
        extended = distgraph.min_cost(0, indirect=True)
        self.assertEqual(expected_extended, list(extended), 'Unexpected indirect min-cost vector after update')

        distgraph.lighten_edge(1, 11, 10)
        expected_extended = [0.0, 20.0, 34.0, 48.0, 50.0, 26.0, 36.0, 50.0, 26.0, 35.0, 36.0, 20.0, 52.0, 52.0, 52.0,
                             35.0, 63.0, 34.0, 34.0, 108.0, 106.0, 79.0, 35.0, 75.0, 69.0, 103.0, 49.0, 56.0, 50.0, 104.0,
                             148.0, 53.0, 53.0, 50.0, 50.0, 74.0, 70.0]
        extended = distgraph.min_cost(0, indirect=True)
        self.assertEqual(expected_extended, list(extended), 'Unexpected indirect min-cost vector after update')

    def test_min_cost_in_single_component_with_distances(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        galaxy, graph, _, _ = self._setup_graph(sourcefile)
        components = galaxy.trade.components
        self.assertEqual(1, len(components))

        distgraph = DistanceGraph(graph, use_distances=True)
        expected = [0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0]
        actual = distgraph.min_cost(0)
        self.assertEqual(expected, list(actual), 'Unexpected min-cost vector')

        expected_extended = [0.0, 3.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                             2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                             3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        extended = distgraph.min_cost(0, indirect=True)
        self.assertEqual(expected_extended, list(extended), 'Unexpected indirect min-cost vector')

    def test_min_cost_in_multiple_components(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar.sec')
        galaxy, graph, _, _ = self._setup_graph(sourcefile)
        components = galaxy.trade.components
        self.assertEqual(8, len(components))

        distgraph = DistanceGraph(graph)

        expected = [0, 70, 48, 48, 50, 26, 51, 50, 26, 50, 50, 51, 52, 52, 52, 49, 77, 48, 48, 108,
                    106, 79, 49, 75, 69, 103, 49, 54, 49, 104, 144, 53, 52, 49, 48, 73, 69, 52, 48, 72,
                    52, 108, 78, 71, 78, 51, 51, 54, 81, 54, 54, 54, 104, 51, 73, 55, 51, 80, 105, 51,
                    48, 48, 54, 54, 74, 54, 52, 54, 54, 108, 71, 107, 72, 81, 51, 51, 52, 52, 52, 106,
                    50, 105, 51, 52, 50, 49, 48, 52, 107, 51, 48, 52, 73, 77, 52, 54, 54, 69, 74, 106,
                    74, 83, 72, 106, 51, 51, 52, 52, 52, 50, 50, 52, 104, 50, 50, 49, 50, 53, 52, 49,
                    50, 48, 105, 53, 49, 50, 48, 48, 50, 52, 48, 49, 48, 49, 69, 48, 78, 124, 80, 52,
                    52, 51, 72, 51, 48, 126, 106, 79, 49, 128, 71, 54, 54, 52, 79, 48, 108, 48, 109, 74,
                    102, 104, 54, 54, 120, 0, 54, 80, 54, 75, 54, 54, 108, 182, 108, 109, 74, 54, 52, 49,
                    108, 54, 52, 49, 51, 83, 53, 54, 51, 52, 81, 106, 53, 52, 53, 52, 55, 72, 52, 52,
                    53, 71, 79, 73, 49, 53, 56, 79, 80, 52, 49, 48, 52, 53, 48, 49, 104, 56, 56, 53,
                    51, 51, 104, 72, 53, 50, 52, 52, 108, 104, 50, 78, 75, 80, 55, 103, 104, 107, 53, 71,
                    53, 46, 102, 49, 80, 54, 50, 46, 100, 52, 52, 54, 80, 49, 52, 52, 54, 54, 105, 49,
                    52, 52, 54, 55, 54, 80, 104, 55, 80, 52, 76, 55, 52, 48, 51, 107, 72, 52, 54, 50,
                    82, 105, 52, 76, 50, 53, 54, 51, 102, 75, 79, 78, 54, 0, 48, 48, 53, 53, 53, 51,
                    49, 54, 53, 51, 76, 51, 55, 54, 52, 70, 79, 106, 51, 104, 77, 109, 52, 52, 106, 106,
                    73, 50, 50, 52, 50, 50, 50, 51, 50, 78, 52, 79, 52, 54, 52, 52, 52, 53, 108, 55,
                    73, 53, 55, 109, 52, 54, 56, 82, 74, 52, 56, 108, 73, 54, 53, 106, 52, 52, 52, 53,
                    108, 53, 102, 50, 79, 52, 51, 79, 0, 50, 51, 51, 71, 101, 51, 50, 50, 94, 78, 44,
                    54, 69, 103, 46, 55, 104, 71, 54, 80, 51, 49, 54, 107, 54, 54, 49, 50, 55, 49, 48,
                    48, 108, 107, 72, 80, 108, 102, 50, 0, 94, 88, 88, 133, 170, 88, 0, 88, 73, 73, 54,
                    54, 99, 81, 74, 54, 54, 54, 50, 50, 103, 50, 54, 54, 106, 74, 108, 55, 55, 107, 54,
                    54, 106, 52, 75, 109, 56, 56, 51, 53, 48, 50, 53, 81, 52, 49, 48, 78, 71, 48, 47,
                    77, 50, 49, 52, 52, 49, 44, 52, 80, 53, 54, 104, 45, 47, 96, 70, 50, 72, 69, 121,
                    122, 50, 76, 94, 125, 97, 90, 52, 74, 108, 92, 84, 124, 129, 120, 90]
        actual = distgraph.min_cost(0, indirect=True)
        self.assertEqual(expected, list(actual), 'Unexpected indirect min-cost vector')

    def test_ordering_of_node(self):
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar.sec')
        galaxy, graph, _, _ = self._setup_graph(sourcefile)
        components = galaxy.trade.components
        self.assertEqual(8, len(components))

        distgraph = DistanceGraph(graph)
        expected = (np.array([2, 6, 10, 11], dtype=np.int64), np.array([46.0, 48.0, 48, 48], dtype=np.float64), {2: 0, 6: 1, 10: 2, 11: 3}, np.array([2, 2, 2, 2], dtype=np.float64))
        actual = distgraph._arcs[1]
        self.assertEqual(len(expected), len(actual))

        for i in range(len(expected)):
            if isinstance(expected[i], np.ndarray):
                self.assertEqual(list(expected[i]), list(actual[i]))
            else:
                self.assertEqual(expected[i], actual[i])

        distgraph.lighten_edge(1, 10, 24)
        expected = (np.array([2, 6, 10, 11], dtype=np.int64), np.array([46.0, 48.0, 24, 48], dtype=np.float64), {2: 0, 6: 1, 10: 2, 11: 3}, np.array([2, 2, 2, 2], dtype=np.float64))
        actual = distgraph._arcs[1]

        for i in range(len(expected)):
            if isinstance(expected[i], np.ndarray):
                self.assertEqual(list(expected[i]), list(actual[i]))
            else:
                self.assertEqual(expected[i], actual[i])

    def _setup_graph(self, sourcefile):
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector
        args = self._make_args()
        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, False)
        galaxy.output_path = args.output
        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        graph = galaxy.stars
        stars = list(graph.nodes)
        source = stars[0]
        return galaxy, graph, source, stars
