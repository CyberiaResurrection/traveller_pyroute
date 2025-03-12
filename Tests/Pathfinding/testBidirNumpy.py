"""
Created on Mar 11, 2025

@author: CyberiaResurrection
"""
import pytest

from PyRoute.AreaItems.Galaxy import Galaxy
from PyRoute.DataClasses.ReadSectorOptions import ReadSectorOptions
from Tests.baseTest import baseTest
goodimport = True

try:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnified import ApproximateShortestPathForestUnified
except ModuleNotFoundError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
except ImportError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified

try:
    from PyRoute.Pathfinding.bidir_numpy import bidir_path_numpy
except ModuleNotFoundError:
    goodimport = False
except ImportError:
    goodimport = False


class testBidirNumpy(baseTest):

    def setUp(self) -> None:
        if not goodimport:
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

        stardex = 13
        targdex = 29
        upbound = 112.6
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
