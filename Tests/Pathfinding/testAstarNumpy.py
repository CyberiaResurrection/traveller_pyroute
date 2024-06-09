"""
Created on Sep 21, 2023

@author: CyberiaResurrection
"""
from PyRoute.Pathfinding.ApproximateShortestPathTreeDistanceGraph import ApproximateShortestPathTreeDistanceGraph
from PyRoute.Pathfinding.DistanceGraph import DistanceGraph
from PyRoute.DeltaDebug.DeltaDictionary import SectorDictionary, DeltaDictionary
from PyRoute.DeltaDebug.DeltaGalaxy import DeltaGalaxy
from Tests.baseTest import baseTest
from PyRoute.Pathfinding.astar_numpy import astar_path_numpy
from PyRoute.Pathfinding.bidirectional_astar_numpy import bidirectional_astar_path_numpy


class testAStarNumpy(baseTest):

    def testAStarOverSubsector(self):
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        args = self._make_args()

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        dist_graph = DistanceGraph(galaxy.stars)

        source = galaxy.star_mapping[0]
        target = galaxy.star_mapping[36]

        galaxy.trade.shortest_path_tree = ApproximateShortestPathTreeDistanceGraph(source.index, galaxy.stars, 0)

        heuristic = galaxy.heuristic_distance_bulk

        exp_route = [0, 8, 9, 15, 24, 36]
        exp_diagnostics = {'branch_factor': 1.704, 'f_exhausted': 3, 'g_exhausted': 3, 'neighbour_bound': 14,
                            'new_upbounds': 1, 'nodes_expanded': 16, 'nodes_queued': 18, 'nodes_revisited': 1,
                            'num_jumps': 5, 'un_exhausted': 7, 'targ_exhausted': 1}
        exp_cost = 239.0

        upbound = galaxy.trade.shortest_path_tree.triangle_upbound(source, target) * 1.005
        act_route, diagnostics = astar_path_numpy(dist_graph, source.index, target.index, heuristic, upbound=upbound, diagnostics=True)
        act_cost = galaxy.route_cost(exp_route)
        self.assertEqual(exp_route, act_route)
        self.assertEqual(exp_cost, act_cost)
        self.assertEqual(exp_diagnostics, diagnostics)

    def testBidirectionalAStarOverSubsector(self):
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        args = self._make_args()

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        dist_graph = DistanceGraph(galaxy.stars)
        heuristic = dist_graph.distances_from_target

        source = galaxy.star_mapping[0]
        target = galaxy.star_mapping[36]

        exp_route = [0, 8, 9, 15, 24, 36]
        exp_diagnostics = {'branch_factor': 2.0, 'f_exhausted': 9, 'g_exhausted': 1, 'neighbour_bound': 19,
                            'new_upbounds': 2, 'nodes_expanded': 21, 'nodes_queued': 30, 'nodes_revisited': 1,
                            'num_jumps': 5, 'un_exhausted': 9, 'targ_exhausted': 0}
        exp_cost = 239.0
        act_route, diagnostics = bidirectional_astar_path_numpy(dist_graph, source.index, target.index, heuristic)
        act_cost = galaxy.route_cost(exp_route)
        self.assertEqual(exp_route, act_route)
        self.assertEqual(exp_cost, act_cost)
        self.assertEqual(exp_diagnostics, diagnostics)

    def testBidirectionalAStarOverSubsectorWithBulkHeuristic(self):
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        args = self._make_args()

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output
        galaxy.bidir_path = True

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        galaxy.trade.shortest_path_tree = ApproximateShortestPathTreeDistanceGraph(0, galaxy.stars, 0)
        dist_graph = DistanceGraph(galaxy.stars)
        heuristic = galaxy.heuristic_distance_bulk

        source = galaxy.star_mapping[0]
        target = galaxy.star_mapping[36]

        exp_route = [0, 8, 9, 15, 24, 36]
        exp_diagnostics = {'branch_factor': 1.979, 'f_exhausted': 6, 'g_exhausted': 0, 'neighbour_bound': 17,
                            'new_upbounds': 1, 'nodes_expanded': 19, 'nodes_queued': 29, 'nodes_revisited': 1,
                            'num_jumps': 5, 'un_exhausted': 11, 'targ_exhausted': 0}
        exp_cost = 239.0
        act_route, diagnostics = bidirectional_astar_path_numpy(dist_graph, source.index, target.index, heuristic)
        act_cost = galaxy.route_cost(exp_route)
        self.assertEqual(exp_route, act_route)
        self.assertEqual(exp_cost, act_cost)
        self.assertEqual(exp_diagnostics, diagnostics)

    def testBidirectionalAStarOverSubsectorSourceAndTargetConnected(self):
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        args = self._make_args()

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        dist_graph = DistanceGraph(galaxy.stars)
        heuristic = dist_graph.distances_from_target

        source = galaxy.star_mapping[0]
        target = galaxy.star_mapping[8]

        exp_route = [0, 8]
        exp_diagnostics = {'branch_factor': 4.0, 'f_exhausted': 3, 'g_exhausted': 0, 'neighbour_bound': 3,
                            'new_upbounds': 1, 'nodes_expanded': 4, 'nodes_queued': 4, 'nodes_revisited': 0,
                            'num_jumps': 1, 'un_exhausted': 0, 'targ_exhausted': 0}
        exp_cost = 74.0
        act_route, diagnostics = bidirectional_astar_path_numpy(dist_graph, source.index, target.index, heuristic)
        act_cost = galaxy.route_cost(exp_route)
        self.assertEqual(exp_route, act_route)
        self.assertEqual(exp_cost, act_cost)
        self.assertEqual(exp_diagnostics, diagnostics)

    def testBidirectionalAStarBoundBlowup(self):
        sourcefile = self.unpack_filename('DeltaFiles/bidirectional_astar_blowups/Zdiedeiant.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        args = self._make_args()

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        galaxy.trade.shortest_path_tree = ApproximateShortestPathTreeDistanceGraph(0, galaxy.stars, 0)
        dist_graph = DistanceGraph(galaxy.stars)
        heuristic = galaxy.heuristic_distance_bulk

        galaxy.trade.calculate_routes()
