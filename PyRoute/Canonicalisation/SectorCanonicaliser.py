"""
Created on Nov 16, 2023

@author: CyberiaResurrection

A wrapper to enable a supplied sector to be converted (as far as possible) to a canonical form.

"""

import requests

from PyRoute.DeltaStar import DeltaStar
from PyRoute.DeltaDebug.DeltaDictionary import SectorDictionary


class SectorCanonicaliser(object):

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
        self.dictionary.canonicalise_headers()

        self.dictionary = self.dictionary.switch_lines(self.processed_lines)
        self.processed_lines = []
        self.dictionary.write_file(self.output_dir, suffix='-canonical')

    def lintsec(self, url):
        data = []
        for line in self.dictionary.headers:
            data.append(line)
        for line in self.dictionary.lines:
            data.append(line)
        headers = {'Content-Type': 'text/plain; charset=utf-8'}
        data = '\n'.join(data)

        r = requests.post(url, data, headers=headers)
        if 200 == r.status_code:
            return True, []

        if 400 == r.status_code:
            content = r.text
            lines = content.splitlines()
            warnings = [item for item in lines if 'Warning:' in item]

            return False, warnings
