"""
Created on Dec 21, 2025

@author: CyberiaResurrection
"""
import logging

from numpy import dtype

from PyRoute import Star
from PyRoute.AreaItems.Galaxy import Galaxy
from PyRoute.AreaItems.Sector import Sector
from PyRoute.DataClasses.ReadSectorOptions import ReadSectorOptions
from PyRoute.Inputs.ParseStarInput import ParseStarInput
from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
from PyRoute.Pathfinding.DistanceGraph import DistanceGraph
from Tests.baseTest import baseTest


class testApproximateShortestPathForestUnifiedFallback(baseTest):

    def setUp(self) -> None:
        ParseStarInput.deep_space = {}
        logger = logging.getLogger('PyRoute.Star')
        logger.setLevel(50)
        logger = logging.getLogger('PyRoute.Galaxy')
        logger.setLevel(50)
        logger = logging.getLogger('PyRoute.TradeCalculation')
        logger.setLevel(50)

    def test_init_1(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        galaxy.trade.star_graph = DistanceGraph(galaxy.stars)

        shortest_path_tree = ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources={0: 1})
        self.assertEqual(0, shortest_path_tree._source)
        self.assertEqual({0: 1}, shortest_path_tree._sources)
        self.assertEqual([[1]], shortest_path_tree._seeds)
        self.assertEqual(0.1, shortest_path_tree._epsilon)
        self.assertEqual(1 / 1.1, shortest_path_tree._divisor)
        self.assertEqual(1, shortest_path_tree._num_trees)
        self.assertEqual(37, shortest_path_tree._graph_len)

        self.assertEqual(dtype('float64'), shortest_path_tree._distances.dtype)
        self.assertEqual(dtype('float64'), shortest_path_tree._max_labels.dtype)

        distances = shortest_path_tree.distances.round(6)
        distances = distances.tolist()
        self.assertEqual([158.181824], distances[0])
        self.assertEqual([0.0], distances[1])
        self.assertEqual([41.818184], distances[2])
        self.assertEqual([64.545456], distances[3])
        self.assertEqual([83.636368], distances[4])
        self.assertEqual([109.090912], distances[5])
        self.assertEqual([43.636364], distances[6])
        self.assertEqual([83.636368], distances[7])

    def test_init_2(self) -> None:
        galaxy = Galaxy(min_btn=15, max_jump=4)
        exp_msg = 'Source node # -1 not in source graph'
        msg = None

        try:
            ApproximateShortestPathForestUnified(-1, galaxy.stars, 0.1, sources=None)
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_init_3(self) -> None:
        galaxy = Galaxy(min_btn=15, max_jump=4)
        exp_msg = 'Source node Foostar (Core None) has undefined component.  Has calculate_components() been run?'
        msg = None

        sector = Sector('# Core', '# 0, 0')
        source = Star()
        source.sector = sector
        source.name = "Foostar"

        try:
            ApproximateShortestPathForestUnified(source, galaxy.stars, 0.1, sources=None)
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_init_4(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()

        exp_msg = 'Source node Didraga (Zarushagar 0101) has undefined component.  Has calculate_components() been run?'
        msg = None
        try:
            ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources={0: 0})
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_init_5(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        del galaxy.stars.nodes[0]['star']

        exp_msg = 'Source node # 0 does not have star attribute'
        msg = None
        try:
            ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources={0: 0})
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_init_6(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()

        exp_msg = 'Source node # -1 not in source graph'
        msg = None
        try:
            ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources=[-1])
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_init_7(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        galaxy.trade.star_graph = DistanceGraph(galaxy.stars)

        shortest_path_tree = ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1)
        self.assertEqual(0, shortest_path_tree._source)
        self.assertEqual(None, shortest_path_tree._sources)
        self.assertEqual([[0]], shortest_path_tree._seeds)
        self.assertEqual(0.1, shortest_path_tree._epsilon)
        self.assertEqual(1 / 1.1, shortest_path_tree._divisor)
        self.assertEqual(1, shortest_path_tree._num_trees)
        self.assertEqual(37, shortest_path_tree._graph_len)

        self.assertEqual(dtype('float64'), shortest_path_tree._distances.dtype)
        self.assertEqual(dtype('float64'), shortest_path_tree._max_labels.dtype)

        distances = shortest_path_tree.distances.round(6)
        distances = distances.tolist()
        self.assertEqual([0.0], distances[0])
        self.assertEqual([158.181824], distances[1])
        self.assertEqual([200.0], distances[2])
        self.assertEqual([222.72728], distances[3])
        self.assertEqual([241.818176], distances[4])
        self.assertEqual([49.090912], distances[5])
        self.assertEqual([114.545456], distances[6])
        self.assertEqual([241.818176], distances[7])

    def test_init_8(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()

        sector = Sector('# Core', '# 0, 0')
        source = Star()
        source.sector = sector
        source.name = "Foostar"

        exp_msg = 'Source node Foostar (Core None) has undefined component.  Has calculate_components() been run?'
        msg = None
        try:
            ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources=[source])
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_init_9(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()

        sector = Sector('# Core', '# 0, 0')
        source = Star()
        source.sector = sector
        source.name = "Foostar"

        del galaxy.stars.nodes[1]['star']

        exp_msg = 'Source node # 1 does not have star attribute'
        msg = None
        try:
            ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources=[1])
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_init_10(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=4)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()

        sector = Sector('# Core', '# 0, 0')
        source = Star()
        source.sector = sector
        source.name = "Foostar"

        galaxy.stars.nodes[1]['star'].component = None

        exp_msg = 'Source node Ymirial (Zarushagar 0106) has undefined component.  Has calculate_components() been run?'
        msg = None
        try:
            ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources=[1])
        except ValueError as e:
            msg = str(e)
        self.assertEqual(exp_msg, msg)

    def test_lower_bound_bulk_1(self) -> None:
        args = self._make_args()
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        readparms = ReadSectorOptions(sectors=[sourcefile], pop_code=args.pop_code, ru_calc=args.ru_calc,
                                      route_reuse=args.route_reuse, trade_choice=args.routes, route_btn=args.route_btn,
                                      mp_threads=args.mp_threads, debug_flag=args.debug_flag, fix_pop=False,
                                      deep_space={}, map_type=args.map_type)

        galaxy = Galaxy(min_btn=15, max_jump=1)
        galaxy.read_sectors(readparms)

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        landmarks, component_landmarks = galaxy.trade.get_landmarks()
        shortest_path_tree = ApproximateShortestPathForestUnified(0, galaxy.stars, 0.1, sources=landmarks)
        lobound = shortest_path_tree.lower_bound_bulk(2)
        self.assertEqual(418.1817855834961, lobound[0])
        self.assertEqual(float('inf'), lobound[1])
        self.assertEqual(0.0, lobound[2])
        self.assertEqual(21.81818389892578, lobound[3])
        self.assertEqual(45.45454788208008, lobound[4])
        self.assertEqual(369.0908737182617, lobound[5])

        lobound = shortest_path_tree.lower_bound_bulk(1)
        self.assertEqual(dtype('float64'), lobound.dtype)
        self.assertEqual(0.0, lobound[0])
        self.assertEqual(0.0, lobound[1])
        self.assertEqual(0.0, lobound[2])
        self.assertEqual(0.0, lobound[3])
        self.assertEqual(0.0, lobound[4])
        self.assertEqual(0.0, lobound[5])
