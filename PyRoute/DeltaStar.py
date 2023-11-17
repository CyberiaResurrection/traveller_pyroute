"""
Created on Jun 13, 2023

@author: CyberiaResurrection
"""
import logging
from typing import Optional

from PyRoute.AreaItems.Sector import Sector
from PyRoute.Inputs.ParseStarInput import ParseStarInput
from PyRoute.Nobles import Nobles
from PyRoute.Star import Star
from PyRoute.TradeCodes import TradeCodes
from SystemData.Utilities import Utilities


class DeltaStar(Star):

    @staticmethod
    def reduce(starline, drop_routes=False, drop_trade_codes=False, drop_noble_codes=False, drop_base_codes=False,
               drop_trade_zone=False, drop_extra_stars=False, reset_pbg=False, reset_worlds=False, reset_port=False,
               reset_tl=False, reset_sophont=False, reset_capitals=False, canonicalise=False, trim_noble_codes=False,
               trim_trade_codes=False, trim_base_codes=False, trim_trade_zone=False) -> Optional[str]:
        sector = Sector("# dummy", "# 0, 0")
        star = DeltaStar.parse_line_into_star(starline, sector, 'fixed', 'fixed')
        if not isinstance(star, DeltaStar):
            return None

        if canonicalise:
            star.canonicalise()
        if drop_routes:
            star.reduce_routes()
        if drop_trade_codes:
            star.reduce_trade_codes()
        elif trim_trade_codes:
            star.trim_trade_codes()
        if drop_noble_codes:
            star.reduce_noble_codes()
        elif trim_noble_codes:
            star.trim_noble_codes()
        if drop_base_codes:
            star.reduce_base_codes()
        elif trim_base_codes:
            star.trim_base_codes()
        if drop_trade_zone:
            star.reduce_trade_zone()
        elif trim_trade_zone:
            star.trim_trade_zone()
        if drop_extra_stars:
            star.reduce_extra_stars()
        if reset_pbg:
            star.reduce_pbg()
        if reset_worlds:
            star.reduce_worlds()
        if reset_port:
            star.reduce_port()
        if reset_tl:
            star.reduce_tl()
        if reset_sophont:
            star.reduce_sophonts()
        if reset_capitals:
            star.reduce_capitals()

        star.calculate_importance()
        return star.parse_to_line()

    @staticmethod
    def reduce_all(original) -> Optional[str]:
        return DeltaStar.reduce(original, drop_routes=True, drop_trade_codes=True, drop_noble_codes=True, drop_base_codes=True, drop_trade_zone=True, drop_extra_stars=True, reset_pbg=True, reset_worlds=True, reset_port=True, reset_tl=True, reset_sophont=True, canonicalise=False)

    @staticmethod
    def reduce_auxiliary(original) -> Optional[str]:
        return DeltaStar.reduce(original, drop_routes=True, drop_noble_codes=True, drop_trade_zone=True, drop_extra_stars=True, reset_pbg=True, reset_worlds=True, reset_sophont=True)

    @staticmethod
    def reduce_importance(original) -> Optional[str]:
        return DeltaStar.reduce(original, drop_trade_codes=True, drop_base_codes=True, reset_port=True, reset_tl=True)

    @staticmethod
    def reduce_nbz(original) -> Optional[str]:
        return DeltaStar.reduce(original, drop_noble_codes=True, drop_base_codes=True, drop_trade_zone=True)

    @staticmethod
    def parse_line_into_star(line, sector, pop_code, ru_calc, fix_pop=False, fix_econ=False):
        star = DeltaStar()
        return ParseStarInput.parse_line_into_star_core(star, line, sector, pop_code, ru_calc, fix_pop=fix_pop, fix_econ=fix_econ)

    def reduce_routes(self) -> None:
        self.routes = []

    def reduce_trade_codes(self) -> None:
        self.tradeCode = TradeCodes('')  # pragma: no mutate

    def trim_trade_codes(self) -> None:
        matches = {"In", "Ni", "Ag", "Ex", "Hi", "Ri", "Cp", "Cs", "Cx"}.union(TradeCodes.ex_codes)
        trade_string = ""  # pragma: no mutate
        for codes in self.tradeCode.codes:
            if codes in matches:
                trade_string += " " + str(codes)

        self.tradeCode = TradeCodes(trade_string)

    def reduce_noble_codes(self) -> None:
        self.nobles = Nobles()

    def trim_noble_codes(self) -> None:
        noble_code = str(self.nobles)
        if 2 > len(noble_code):
            return

        self.nobles = Nobles()
        self.nobles.count(noble_code[0])

    def reduce_base_codes(self) -> None:
        self.baseCode = '-'

    def trim_base_codes(self) -> None:
        base_code = str(self.baseCode)
        if 2 > len(base_code):
            return

        self.baseCode = base_code[0]

    def reduce_trade_zone(self) -> None:
        self.zone = '-'

    def trim_trade_zone(self) -> None:
        if 'R' == self.zone.upper():
            self.zone = 'A'
        elif 'F' == self.zone.upper():
            self.zone = 'U'
        else:
            self.zone = '-'

    def reduce_extra_stars(self) -> None:
        if 0 < len(self.star_list):  # pragma: no mutate
            self.star_list_object.stars_list = [self.star_list_object.stars_list[0]]

    def reduce_pbg(self) -> None:
        self.popM = 1
        self.ggCount = 0
        self.belts = 0

    def reduce_worlds(self) -> None:
        self.worlds = 1

    def reduce_port(self) -> None:
        self.port = 'C'  # pragma: no mutate

    def reduce_tl(self) -> None:
        self.tl = 8

    def reduce_sophonts(self) -> None:
        nu_codes = []

        for code in self.tradeCode.codes:
            if code not in self.tradeCode.sophont_list and not code.startswith('Di('):
                nu_codes.append(code)

        self.tradeCode = TradeCodes(' '.join(nu_codes))

    def reduce_capitals(self) -> None:
        cap_codes = ['Cs', 'Cp', 'Cx']

        nu_codes = []
        for code in self.tradeCode.codes:
            if code not in cap_codes:
                nu_codes.append(code)

        self.tradeCode = TradeCodes(' '.join(nu_codes))


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))
