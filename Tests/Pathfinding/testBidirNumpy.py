"""
Created on Mar 11, 2025

@author: CyberiaResurrection
"""
import pytest

from PyRoute.AreaItems.Galaxy import Galaxy
from PyRoute.DataClasses.ReadSectorOptions import ReadSectorOptions
from Tests.baseTest import baseTest
good_import = True

try:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnified import ApproximateShortestPathForestUnified
except ModuleNotFoundError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
except ImportError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
try:
    from PyRoute.Pathfinding.bidir_numpy import bidir_path_numpy
except ModuleNotFoundError:
    good_import = False
except ImportError:
    good_import = False


class testBidirNumpy(baseTest):

    def setUp(self) -> None:
        if not good_import:
            pytest.skip("Need bidir pathfinding imported")

    def testBidirectionalBlowup2(self):
        sourcefile = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_2/Dagudashaag.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()

    def testBidirectionalBlowup3(self):
        sourcefile = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_3/Dagudashaag.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()

    def testBidirectionalBlowup4(self):
        sourcefile = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_4/Solomani Rim.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()

    def testBidirectionalBlowup5(self):
        sourcefile = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_5/Zarushagar.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()

    def testBidirectionalBlowup5a(self):
        sourcefile = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_5/Zarushagar.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        source = max(galaxy.star_mapping.values(), key=lambda item: item.wtn)
        source.is_landmark = True
        galaxy.trade.shortest_path_tree = ApproximateShortestPathForestUnified(source.index, galaxy.stars,
                                                                               galaxy.trade.epsilon)

        raw_route, _ = bidir_path_numpy(galaxy.trade.star_graph, 39, 47,
                                        galaxy.trade.shortest_path_tree.lower_bound_bulk)

        self.assertIsNotNone(raw_route, "Candidates should return a route")

    def testBidirectionalBlowup6(self):
        sourcefile = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_6/Zarushagar.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()

    def testBidirectionalBlowup7(self):
        sourcefile1 = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_7/Diaspora.sec')
        sourcefile2 = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_7/Old Expanses.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile1, sourcefile2], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()

    def testBidirectionalBlowup8(self):
        sourcefile1 = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_8/Fornast.sec')
        sourcefile2 = self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_8/Old Expanses.sec')

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=[sourcefile1, sourcefile2], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()

    def testBidirectionalBlowup9(self):
        sources = [
            self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_9/Core.sec'),
            self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_9/Fornast.sec'),
            self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_9/Gateway.sec'),
            self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_9/Ley Sector.sec'),
            self.unpack_filename('DeltaFiles/bidirectional_duplicate_blowup_9/Lishun.sec'),
        ]

        args = self._make_args()
        args.route_btn = 8
        args.max_jump = 4
        args.routes = "trade"
        readparms = ReadSectorOptions(sectors=sources, pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_components()

        galaxy.trade.calculate_routes()
