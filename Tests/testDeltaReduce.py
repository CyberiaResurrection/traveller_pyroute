import argparse
import math
import tempfile
import unittest

from PyRoute.DeltaDebug.DeltaDictionary import SectorDictionary, DeltaDictionary
from PyRoute.DeltaDebug.DeltaGalaxy import DeltaGalaxy
from PyRoute.DeltaDebug.DeltaReduce import DeltaReduce
try:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnified import ApproximateShortestPathForestUnified
except ModuleNotFoundError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
except ImportError:
    from PyRoute.Pathfinding.ApproximateShortestPathForestUnifiedFallback import ApproximateShortestPathForestUnified
from Tests.baseTest import baseTest


class testDeltaReduce(baseTest):
    def test_subsector_reduction(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Dagudashaag-spiked.sec')

        args = self._make_args_no_line()

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_interesting()
        reducer.reduce_subsector_pass()

        self.assertEqual(1, len(reducer.sectors))
        # only one subsector should be non-empty after reduction
        for subsector_name in reducer.sectors['Dagudashaag']:
            expected = 0
            affix = " not empty after subsector reduction"
            if subsector_name == 'Pact':
                expected = 39
                affix = " empty after subsector reduction"
            actual = 0 if reducer.sectors['Dagudashaag'][subsector_name].items is None else len(reducer.sectors['Dagudashaag'][subsector_name].items)
            self.assertEqual(expected, actual, subsector_name + affix)
        # verify sector headers got taken across
        self.assertEqual(len(sector.headers) - 2, len(reducer.sectors['Dagudashaag'].headers), "Unexpected headers length")
        # verify sector allegiances got taken across
        self.assertEqual(
            len(sector.allegiances) - 2,
            len(reducer.sectors['Dagudashaag'].allegiances),
            "Unexpected allegiances length"
        )

    def test_subsector_reduction_allegiance_balance(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/dagudashaag-allegiance-pax-balance/Dagudashaag.sec')

        args = self._make_args_no_line()
        args.interestingline = ": Allegiance total"

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)
        reducer.reduce_line_pass()
        reducer.is_initial_state_uninteresting()

    def test_line_reduction_throws_delta_logic_error(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/dagudashaag-allegiance-pax-balance/Dagudashaag-delta-error.sec')

        args = self._make_args_no_line()
        args.interestingline = ": Allegiance total"

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.reduce_line_pass()
        # input should be un-interesting, so the is-interesting check should blow up with an assertion error
        expected_msg = 'Original input not interesting - aborting'
        msg = None
        try:
            reducer.is_initial_state_interesting()
        except AssertionError as e:
            msg = str(e)
        self.assertEqual(expected_msg, msg)

        # as input should be un-interesting, the is-not-interesting check should not blow up
        reducer.is_initial_state_uninteresting()

    def test_line_reduction(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Dagudashaag-subsector-spiked.sec')

        args = self._make_args()

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_interesting()
        reducer.reduce_line_pass()

        self.assertEqual(1, len(reducer.sectors))
        # only one subsector should be non-empty after reduction
        for subsector_name in reducer.sectors['Dagudashaag']:
            expected = 0
            if subsector_name == 'Pact':
                expected = 2
            self.assertEqual(expected, reducer.sectors['Dagudashaag'][subsector_name].num_lines, subsector_name + " not empty")

        # verify sector headers got taken across
        self.assertEqual(len(sector.headers), len(reducer.sectors['Dagudashaag'].headers), "Unexpected headers length")
        # verify sector allegiances got taken across
        self.assertEqual(
            len(sector.allegiances),
            len(reducer.sectors['Dagudashaag'].allegiances),
            "Unexpected allegiances length"
        )

    def test_line_reduction_singleton_only(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Dagudashaag-subsector-reduced.sec')

        args = self._make_args()

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_interesting()

        # now verify 1-minimality by removing only one line of input at a time
        reducer.reduce_line_pass(singleton_only=True)
        # only one subsector should be non-empty after reduction
        for subsector_name in reducer.sectors['Dagudashaag']:
            expected = 0
            if subsector_name == 'Pact':
                expected = 2
            self.assertEqual(expected, reducer.sectors['Dagudashaag'][subsector_name].num_lines, subsector_name + " not empty")
        self.assertEqual(2, len(reducer.sectors.lines), "Unexpected line count after singleton pass")

    def test_line_reduction_two_minimality(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Dagudashaag-subsector-reduced.sec')

        args = self._make_args()

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_interesting()

        # now verify 2-minimality by removing two lines of input at a time
        reducer.reduce_line_two_minimal()
        # only one subsector should be non-empty after reduction
        for subsector_name in reducer.sectors['Dagudashaag']:
            expected = 0
            if subsector_name == 'Pact':
                expected = 3
            self.assertEqual(expected, reducer.sectors['Dagudashaag'][subsector_name].num_lines, subsector_name + " not empty")
        self.assertEqual(3, len(reducer.sectors.lines), "Unexpected line count after doubleton pass")

    def test_sector_reduction(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Dagudashaag-spiked.sec')

        args = self._make_args()
        args.interestingtype = 'AssertionError'
        args.interestingline = 'duplicated'

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar.sec')
        zarusector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,-1', zarusector.position, "Unexpected position value for Zarushagar")
        delta[zarusector.name] = zarusector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_interesting()
        reducer.reduce_sector_pass()

        self.assertEqual(1, len(reducer.sectors))
        self.assertEqual('Dagudashaag', reducer.sectors['Dagudashaag'].name)

    def test_line_reduction_can_skip_sectors(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Dagudashaag-subsector-spiked.sec')

        args = self._make_args()

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar.sec')
        zarusector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,-1', zarusector.position, "Unexpected position value for Zarushagar")
        delta[zarusector.name] = zarusector

        self.assertEqual(536, len(delta.lines), "Unexpected pre-reduction line count")

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_interesting()
        reducer.reduce_line_pass()

        self.assertEqual(2, len(reducer.sectors.lines), "Unexpected post-reduction line count")
        self.assertEqual(1, len(reducer.sectors), 'Unexpected post-reduction sector count')
        self.assertEqual('Dagudashaag', reducer.sectors['Dagudashaag'].name)

    def test_route_costs_balanced_should_be_uninteresting(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-imbalanced-routes.sec')

        args = self._make_args_no_line()

        delta = DeltaDictionary()
        zarusector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,-1', zarusector.position, "Unexpected position value for Zarushagar")
        delta[zarusector.name] = zarusector

        reducer = DeltaReduce(delta, args)
        reducer.is_initial_state_uninteresting()

    def test_verify_stars_and_shadow_bijection(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/final_result_should_be_independently_interesting/Gushemege.sec')

        args = self._make_args()
        args.btn = 20
        args.max_jump = 4
        args.route_btn = 8
        args.pop_code = 'fixed'
        args.ru_calc = 'scaled'
        args.routes = 'trade'
        args.route_reuse = 10
        args.interestingline = None
        args.interestingtype = 'KeyError'
        args.maps = False
        args.subsectors = False
        args.debugflag = False
        args.mp_threads = 1
        args.run_sector = True
        args.run_subsector = True
        args.run_line = False
        args.run_init = True
        args.two_min = False
        args.borders = 'erode'
        args.ally_match = 'collapse'
        args.owned = False
        args.trade = True
        args.speculative_version = 'CT'
        args.ally_count = 10
        args.json_data = False
        args.output = tempfile.gettempdir()

        delta = DeltaDictionary()
        gushsector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -2,0', gushsector.position, "Unexpected position value for Gushemege")
        delta[gushsector.name] = gushsector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debugflag)

        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.trade.calculate_components()
        galaxy.trade.star_len_root = max(1, math.floor(math.sqrt(len(galaxy.trade.star_graph))) // 2)

        stars = list(galaxy.stars.nodes)

        btn = [(s, n, d) for (s, n, d) in galaxy.ranges.edges(data=True) if s.component == n.component]
        btn.sort(key=lambda tn: tn[2]['btn'], reverse=True)
        galaxy.trade.shortest_path_tree = ApproximateShortestPathForestUnified(stars[0], galaxy.stars, 0.2)

        switch = 7
        line = btn[switch]
        btn = btn[0:switch]

        for (star, neighbour, _) in btn:
            galaxy.trade.get_trade_between(star, neighbour)

        galaxy.trade.get_trade_between(line[0], line[1])

    def test_verify_sector_without_subsector_names_and_generating_maps_is_not_interesting(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh - subsector P.sec')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.subsectors = True
        args.map_type = "classic"

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_uninteresting(reraise=True)

    def test_verify_sector_without_subsector_names_allegiance_balances(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh - subsector P - trimmed.sec')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.subsectors = True
        args.map_type = "classic"

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_uninteresting(reraise=True)

    def test_star_having_no_sector_attribute(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Dagudashaag-star-object-no-sector-attribute.sec')

        args = self._make_args_no_line()

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)
        reducer.is_initial_state_uninteresting(reraise=True)

    def test_population_balance_over_two_sectors(self) -> None:
        args = self._make_args_no_line()
        sourcefile = self.unpack_filename('DeltaFiles/two-sector-pop-balance/Dagudashaag.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        sourcefile = self.unpack_filename('DeltaFiles/two-sector-pop-balance/Zarushagar.sec')

        zarusector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[zarusector.name] = zarusector
        self.assertEqual(2, len(delta), "Should only be two sectors in dictionary")

        reducer = DeltaReduce(delta, args, args.interestingline, args.interestingtype)
        reducer.is_initial_state_uninteresting(reraise=True)

    def test_pax_and_trade_balance_over_reft_sector(self) -> None:
        args = self._make_args_no_line()
        sourcefile = self.unpack_filename('DeltaFiles/reft-allegiance-pax-balance/Reft Sector.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args, args.interestingline, args.interestingtype)
        reducer.is_initial_state_uninteresting(reraise=True)

    def test_zao_kfeng_jump_4_template_blowup(self) -> None:
        args = self._make_args_no_line()
        args.max_jump = 4
        sourcefile = self.unpack_filename('DeltaFiles/zao_kfeng_jump_4_template_blowup/Zao Kfeng Ig Grilokh.sec')

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args, args.interestingline, args.interestingtype)
        reducer.is_initial_state_uninteresting(reraise=True)

    def test_allegiance_reduction(self) -> None:
        sourcefile = self.unpack_filename('DeltaFiles/Passes/Dagudashaag-subsector-full-reduce.sec')

        args = self._make_args()

        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        self.assertEqual('# -1,0', sector.position, "Unexpected position value for Dagudashaag")
        delta = DeltaDictionary()
        delta[sector.name] = sector

        reducer = DeltaReduce(delta, args)

        reducer.is_initial_state_interesting()

        reducer.reduce_allegiance_pass()
        self.assertEqual(1, len(reducer.sectors['Dagudashaag'].allegiances), "Unexpected allegiance count after reduction")
        for alg_name in reducer.sectors['Dagudashaag'].allegiances:
            expected = 0
            self.assertEqual(expected, len(reducer.sectors['Dagudashaag'].allegiances[alg_name].worlds))

        self.assertEqual(4, len(reducer.sectors.lines), "Unexpected line count after allegiance pass")

    def _make_args(self) -> argparse.ArgumentParser:
        args = argparse.ArgumentParser(description='PyRoute input minimiser.')
        args.btn = 8
        args.max_jump = 2
        args.route_btn = 13
        args.pop_code = 'scaled'
        args.ru_calc = 'scaled'
        args.routes = 'trade'
        args.route_reuse = 10
        args.interestingline = "Weight of edge"
        args.interestingtype = None
        args.maps = None
        args.borders = 'range'
        args.ally_match = 'collapse'
        args.owned = False
        args.trade = True
        args.speculative_version = 'CT'
        args.ally_count = 10
        args.json_data = False
        args.output = tempfile.gettempdir()
        args.mp_threads = 1
        args.debug_flag = False
        args.mindir = tempfile.gettempdir()
        args.map_type = "classic"
        return args

    def _make_args_no_line(self) -> argparse.ArgumentParser:
        args = self._make_args()
        args.interestingline = None

        return args


if __name__ == '__main__':
    unittest.main()
