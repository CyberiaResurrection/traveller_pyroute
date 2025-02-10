"""
Created on Sep 21, 2023

@author: CyberiaResurrection
"""
goodimport = True
try:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnified import ApproximateShortestPathForestUnified
except ModuleNotFoundError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
    goodimport = False
except ImportError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
    goodimport = False
from PyRoute.Pathfinding.DistanceGraph import DistanceGraph
from PyRoute.DeltaDebug.DeltaDictionary import SectorDictionary, DeltaDictionary
from PyRoute.DeltaDebug.DeltaGalaxy import DeltaGalaxy
from Tests.baseTest import baseTest
try:
    from PyRoute.Pathfinding.astar_numpy import astar_path_numpy
except ModuleNotFoundError:
    from PyRoute.Pathfinding.astar_numpy_fallback import astar_path_numpy
    goodimport = False
except ImportError:
    from PyRoute.Pathfinding.astar_numpy_fallback import astar_path_numpy
    goodimport = False


class testAStarNumpy(baseTest):

    def testAStarOverSubsector(self):
        self.maxDiff = None
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

        galaxy.trade.shortest_path_tree = ApproximateShortestPathForestUnified(source.index, galaxy.stars, 0)

        heuristic = galaxy.heuristic_distance_bulk

        exp_route = [0, 8, 9, 15, 24, 36]
        if goodimport:
            exp_diagnostics = {'branch_factor': 1.789, 'f_exhausted': 0, 'g_exhausted': 6, 'neighbour_bound': 19,
                            'new_upbounds': 1, 'nodes_expanded': 22, 'nodes_queued': 21, 'nodes_revisited': 2,
                            'num_jumps': 5, 'un_exhausted': 11, 'targ_exhausted': 2}
        else:
            exp_diagnostics = {'branch_factor': 1.762, 'f_exhausted': 0, 'g_exhausted': 3, 'neighbour_bound': 14,
                               'new_upbounds': 1, 'nodes_expanded': 16, 'nodes_queued': 20, 'nodes_revisited': 1,
                               'num_jumps': 5, 'targ_exhausted': 1, 'un_exhausted': 10}

        upbound = galaxy.trade.shortest_path_tree.triangle_upbound(source.index, target.index) * 1.005
        act_route, diagnostics = astar_path_numpy(dist_graph, source.index, target.index, heuristic, upbound=upbound,
                                                  diagnostics=True)
        self.assertEqual(exp_route, act_route)
        self.assertEqual(exp_diagnostics, diagnostics)
