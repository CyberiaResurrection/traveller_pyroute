import unittest

from PyRoute.Calculation.TradeCalculation import TradeCalculation
from PyRoute.TradeCodes import TradeCodes

from PyRoute.AreaItems.Galaxy import Galaxy
from PyRoute.AreaItems.Sector import Sector
from PyRoute.Inputs.ParseStarInput import ParseStarInput
from PyRoute.Star import Star


class testTradeCalculation(unittest.TestCase):

    def setUp(self) -> None:
        ParseStarInput.deep_space = {}

    def test_negative_route_weight_trips_assertion(self) -> None:
        expected = 'Weight of edge between Irkigkhan (Core 0103) and Irkigkhan (Core 0103) must be positive'
        actual = None
        try:
            sector = Sector('# Core', '# 0, 0')
            star1 = Star.parse_line_into_star(
                "0103 Irkigkhan            B9C4733-9 Fl                   { 0 }  (E69+0) [4726] B     - - 123 8  Im M2 V           ",
                sector, 'fixed', 'fixed')

            galaxy = Galaxy(min_btn=13)
            tradecalc = TradeCalculation(galaxy)

            tradecalc.route_weight(star1, star1)
        except AssertionError as e:
            actual = str(e)
            pass

        self.assertEqual(expected, actual)

    def test_zero_route_weight_trips_assertion(self) -> None:
        expected = 'Weight of edge between Irkigkhan (Core 0103) and Irkigkhan (Core 0103) must be positive'
        actual = None
        try:
            sector = Sector('# Core', '# 0, 0')
            star1 = Star.parse_line_into_star(
                "0103 Irkigkhan            B9C4733-9 Fl                   { 0 }  (E69+0) [4726] B     - - 123 8  Im M2 V           ",
                sector, 'fixed', 'fixed')
            star1.importance = 0

            galaxy = Galaxy(min_btn=13)
            tradecalc = TradeCalculation(galaxy)

            tradecalc.route_weight(star1, star1)
        except AssertionError as e:
            actual = str(e)
            pass

        self.assertEqual(expected, actual)

    def test_positive_route_weight_doesnt_trip_assertion(self) -> None:
        sector = Sector('# Core', '# 0, 0')
        star1 = Star.parse_line_into_star(
            "0103 Irkigkhan            B9C4733-9 Fl                   { 0 }  (E69+0) [4726] B     - - 123 8  Im M2 V           ",
            sector, 'fixed', 'fixed')
        star1.importance = -1

        galaxy = Galaxy(min_btn=13)
        tradecalc = TradeCalculation(galaxy)

        self.assertEqual(2, tradecalc.route_weight(star1, star1))

    def test_single_link_route_weight(self) -> None:
        sector = Sector('# Core', '# 0, 0')
        star1 = Star.parse_line_into_star(
            "0103 Irkigkhan            B9C4733-9 Fl                   { 0 }  (E69+0) [4726] B     - - 123 8  Im M2 V           ",
            sector, 'fixed', 'fixed')
        star1.index = 1
        star2 = Star.parse_line_into_star(
            "0104 Shana Ma             E551112-7 Lo Po                { -3 } (301-3) [1113] B     - - 913 9  Im K2 IV M7 V     ",
            Sector('# Core', '# 0, 0'), 'fixed', 'fixed')
        star2.index = 2

        galaxy = Galaxy(min_btn=13)
        tradecalc = TradeCalculation(galaxy)

        route = [star1, star2]
        galaxy.stars.add_node(star1.index, star=star1)
        galaxy.stars.add_node(star2.index, star=star2)
        galaxy.ranges.add_node(star1)
        galaxy.ranges.add_node(star2)
        galaxy.stars.add_edge(star1.index, star2.index, distance=1,
                              weight=10, trade=0, btn=10, count=0)

        expected_weight = 10
        actual_weight = tradecalc.route_cost(route)
        self.assertEqual(expected_weight, actual_weight, "Route cost unexpected")

    def test_passenger_balance_over_multiple_sectors(self) -> None:
        core = Sector('# Core', '# 0, 0')
        dagu = Sector('# Dagudashaag', '# -1, 0')
        gush = Sector('# Gushemege', '# -2, 0')

        galaxy = Galaxy(min_btn=13)
        galaxy.stats.passengers = 3
        galaxy.sectors[core.name] = core
        galaxy.sectors[dagu.name] = dagu
        galaxy.sectors[gush.name] = gush

        tradecalc = TradeCalculation(galaxy)
        tradecalc.sector_passenger_balance[(core.name, dagu.name)] = 1
        tradecalc.sector_passenger_balance[(core.name, gush.name)] = 1
        tradecalc.sector_passenger_balance[(dagu.name, gush.name)] = 1

        expected = "Uncompensated multilateral passenger imbalance present in sectors"
        actual = None

        try:
            tradecalc.is_sector_pass_balanced()
        except AssertionError as e:
            actual = str(e)

        self.assertEqual(expected, actual, "AssertionError should be thrown")

        tradecalc.multilateral_balance_pass()
        tradecalc.is_sector_pass_balanced()

    def test_trade_balance_over_multiple_sectors(self) -> None:
        core = Sector('# Core', '# 0, 0')
        dagu = Sector('# Dagudashaag', '# -1, 0')
        gush = Sector('# Gushemege', '# -2, 0')

        galaxy = Galaxy(min_btn=13)
        galaxy.stats.trade = 3
        galaxy.sectors[core.name] = core
        galaxy.sectors[dagu.name] = dagu
        galaxy.sectors[gush.name] = gush

        tradecalc = TradeCalculation(galaxy)
        tradecalc.sector_trade_balance[(core.name, dagu.name)] = 1
        tradecalc.sector_trade_balance[(core.name, gush.name)] = 1
        tradecalc.sector_trade_balance[(dagu.name, gush.name)] = 1

        expected = "Uncompensated multilateral trade imbalance present in sectors"
        actual = None

        try:
            tradecalc.is_sector_trade_balanced()
        except AssertionError as e:
            actual = str(e)

        self.assertEqual(expected, actual, "AssertionError should be thrown")

        tradecalc.multilateral_balance_trade()
        tradecalc.is_sector_trade_balanced()

    def test_max_dist(self) -> None:
        galaxy = Galaxy(min_btn=13)
        tradecalc = TradeCalculation(galaxy)
        self.assertEqual(8, tradecalc.min_wtn)

        cand_wtn = [
            (8, 2),
            (9, 9),
            (10, 29),
            (11, 59),
            (12, 99),
            (13, 299),
            (14, 599),
            (15, 599),
        ]

        for wtn, exp_dist in cand_wtn:
            with self.subTest(msg="WTN " + str(wtn)):
                act_dist = tradecalc._max_dist(wtn, wtn)
                self.assertEqual(exp_dist, act_dist)

    def test_max_range_equal_wtn_nothing_fancy(self):
        parmlist = [(6, 0), (7, 2), (8, 9), (9, 29), (10, 99), (11, 299), (12, 999), (13, float('inf'))]
        for wtn, expected in parmlist:
            with self.subTest(wtn=wtn, expected=expected):
                galaxy = Galaxy(0)
                sector = Sector('# Core', '# 0, 0')
                galaxy.sectors[sector.name] = sector

                star1 = self.make_imperial_star1()
                star1.wtn = wtn

                star2 = self.make_imperial_star_2()
                star2.wtn = wtn

                self.assertEqual(2 * wtn, TradeCalculation.get_btn(star1, star2, 0))
                tradecalc = TradeCalculation(galaxy)
                actual = tradecalc.max_route_length(star1, star2)
                self.assertEqual(expected, actual)

    def test_max_range_double_wtn_nothing_fancy(self):
        parmlist = [(6, 29), (7, 199), (8, 999), (9, float('inf')), (10, float('inf'))]

        for wtn, expected in parmlist:
            with self.subTest(wtn=wtn, expected=expected):
                galaxy = Galaxy(0)
                sector = Sector('# Core', '# 0, 0')
                galaxy.sectors[sector.name] = sector

                star1 = self.make_imperial_star1()
                star1.wtn = wtn

                star2 = self.make_imperial_star_2()
                star2.wtn = 2 * wtn

                self.assertEqual(2 * wtn + 1, TradeCalculation.get_btn(star1, star2, 0))
                tradecalc = TradeCalculation(galaxy)
                actual = tradecalc.max_route_length(star1, star2)
                self.assertEqual(expected, actual)

    def test_max_range_at_min_btn_cap(self):
        parmlist = [(7, 1), (8, 2), (9, 5), (10, 9), (11, 19), (12, 29), (13, 59), (14, 99), (15, 199), (16, 299), (17, 599), (18, 999), (19, float('inf'))]

        for wtn, expected in parmlist:
            with self.subTest(wtn=wtn, expected=expected):
                galaxy = Galaxy(0)
                sector = Sector('# Core', '# 0, 0')
                galaxy.sectors[sector.name] = sector

                star1 = self.make_imperial_star1()
                star1.wtn = 6

                star2 = self.make_imperial_star_2()
                star2.wtn = wtn

                self.assertEqual(13, TradeCalculation.get_btn(star1, star2, 0))
                tradecalc = TradeCalculation(galaxy)
                actual = tradecalc.max_route_length(star1, star2)
                self.assertEqual(expected, actual)

    def test_equal_wtn_with_more_reasons_to_trade(self):
        parmlist = [(6, 2), (7, 9), (8, 29), (9, 99), (10, 299), (11, 999), (12, float('inf')), (13, float('inf'))]
        for wtn, expected in parmlist:
            with self.subTest(wtn=wtn, expected=expected):
                galaxy = Galaxy(0)
                sector = Sector('# Core', '# 0, 0')
                galaxy.sectors[sector.name] = sector

                star1 = self.make_imperial_star1()
                star1.wtn = wtn
                star1.tradeCode = TradeCodes("Ag Ni")

                star2 = self.make_imperial_star_2()
                star2.wtn = wtn
                star2.tradeCode = TradeCodes("In Na")

                self.assertEqual(2 * wtn + 1, TradeCalculation.get_btn(star1, star2, 0))
                tradecalc = TradeCalculation(galaxy)
                actual = tradecalc.max_route_length(star1, star2)
                self.assertEqual(expected, actual)
                actual = tradecalc.max_route_length(star2, star1)
                self.assertEqual(expected, actual)

    def test_equal_wtn_with_one_side_wild(self):
        parmlist = [(6, 0), (7, 0), (8, 2), (9, 9), (10, 29), (11, 99), (12, 299), (13, 999), (14, float('inf'))]
        for wtn, expected in parmlist:
            with self.subTest(wtn=wtn, expected=expected):
                galaxy = Galaxy(0)
                sector = Sector('# Core', '# 0, 0')
                galaxy.sectors[sector.name] = sector

                star1 = self.make_imperial_star1()
                star1.wtn = wtn

                star2 = self.make_imperial_star_2()
                star2.wtn = wtn
                star2.alg_code = "Wild"

                self.assertEqual(2 * wtn - 2, TradeCalculation.get_btn(star1, star2, 0))
                tradecalc = TradeCalculation(galaxy)
                actual = tradecalc.max_route_length(star1, star2)
                self.assertEqual(expected, actual)
                actual = tradecalc.max_route_length(star2, star1)
                self.assertEqual(expected, actual)

    def test_equal_wtn_with_differing_allegiance(self):
        parmlist = [(6, 0), (7, 1), (8, 5), (9, 19), (10, 59), (11, 199), (12, 599), (13, float('inf')), (14, float('inf'))]
        for wtn, expected in parmlist:
            with self.subTest(wtn=wtn, expected=expected):
                galaxy = Galaxy(0)
                sector = Sector('# Core', '# 0, 0')
                galaxy.sectors[sector.name] = sector

                star1 = self.make_imperial_star1()
                star1.wtn = wtn

                star2 = self.make_imperial_star_2()
                star2.wtn = wtn
                star2.alg_code = "So"

                self.assertEqual(2 * wtn - 1, TradeCalculation.get_btn(star1, star2, 0))
                tradecalc = TradeCalculation(galaxy)
                actual = tradecalc.max_route_length(star1, star2)
                self.assertEqual(expected, actual)
                actual = tradecalc.max_route_length(star2, star1)
                self.assertEqual(expected, actual)

    @staticmethod
    def make_imperial_star1():
        star1 = Star()
        star1.tradeCode = TradeCodes("")
        star1.alg_code = "Im"
        star1.position = "1009"
        star1.set_location(10, 9)
        return star1

    @staticmethod
    def make_imperial_star_2():
        star2 = Star()
        star2.tradeCode = TradeCodes("")
        star2.alg_code = "Im"
        star2.position = "1010"
        star2.set_location(10, 10)
        return star2


if __name__ == '__main__':
    unittest.main()
