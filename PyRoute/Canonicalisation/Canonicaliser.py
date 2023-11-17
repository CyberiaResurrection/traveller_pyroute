"""
Created on Nov 16, 2023

@author: CyberiaResurrection

A wrapper to enable a supplied sector to be converted (as far as possible) to a canonical form.

"""

from PyRoute.DeltaStar import DeltaStar
from PyRoute.DeltaDebug.DeltaDictionary import SectorDictionary


class Canonicaliser(object):

    def __init__(self, filename, output_dir):
        self.dictionary = SectorDictionary.load_traveller_map_file(filename)
        self.output_dir = output_dir
        self.processed_lines = []

    def process(self):
        self.processed_lines = []
        for line in self.dictionary.lines:
            canon = DeltaStar.reduce(line, canonicalise=True)
            assert isinstance(canon, str), "Candidate line " + line + " was not reduced to a string.  Got " + canon + " instead."
            self.processed_lines.append((line, canon))

    def write(self):
        if not self.processed_lines:
            raise ValueError('Process() has not been run')
        new_hex = 'Hex  Name                 UWP       Remarks                               {Ix}   (Ex)    [Cx]   N    B  Z PBG W  A    Stellar         Routes                                   \n'
        new_dash = '---- -------------------- --------- ------------------------------------- ------ ------- ------ ---- -- - --- -- ---- --------------- -----------------------------------------\n'
        num_headers = len(self.dictionary.headers)
        for i in range(0, num_headers):
            raw_line = self.dictionary.headers[i]
            if raw_line.startswith('Hex  '):
                self.dictionary.headers[i] = new_hex
            elif raw_line.startswith('---- --'):
                self.dictionary.headers[i] = new_dash

        self.dictionary = self.dictionary.switch_lines(self.processed_lines)
        self.processed_lines = []
        self.dictionary.write_file(self.output_dir, suffix='-canonical')
