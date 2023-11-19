"""
Created on Nov 18, 2023

@author: CyberiaResurrection

A wrapper around canonicalising downloaded sector files, running the result
through TravellerMap's lintsec and reporting any warnings.

"""

import argparse
import codecs
import logging
import os

from PyRoute.Canonicalisation.SectorCanonicaliser import SectorCanonicaliser

logger = logging.getLogger('PyRoute')


def get_sectors(sector, input_dir):
    try:
        lines = [line for line in codecs.open(sector, 'r', 'utf-8')]
    except (OSError, IOError):
        logger.error("sector file %s not found" % sector)
    sector_list = []
    for line in lines:
        sector_list.append(os.path.join(input_dir, line.strip() + '.sec'))
    logger.info(sector_list)
    return sector_list


def process():
    parser = argparse.ArgumentParser(description='PyRoute sector canonicalisation.')
    logger.setLevel(logging.CRITICAL)

    top_args = parser.add_argument_group('Arguments', '')
    top_args.add_argument('--input', default='sectors', help='Input directory for sectors to canonicalise.')
    top_args.add_argument('--sectors', default=None, help='File with list of sector names to canonicalise.')
    top_args.add_argument('sector', nargs='*', help='T5SS sector file(s) to process')
    top_args.add_argument('--output-dir', dest="outputdir",
                       help='Output folder to place canonicalised sectors and any warnings generated during canonicalisation.')
    top_args.add_argument('--lintsec-url', dest="lintsec", default='https://travellermap.com/api/sec?lint=1&type=SecondSurvey&hide-tl=0&hide-uwp=0&hide-cap=0',
                      help='Lintsec url to use to check sector canonicalisation')

    args = parser.parse_args()

    raw_sectors_list = args.sector
    if args.sectors is not None:
        raw_sectors_list.extend(get_sectors(args.sectors, args.input))

    sectors_list = []
    for sector in raw_sectors_list:
        if sector not in sectors_list:
            sectors_list.append(sector)
        else:
            logger.warning(sector + " is duplicated")

    counter = 0

    warning_summary = {}

    out_of_range_warnings = {
        'Law': 'Warning: Law=X out of range; should be: Gov(=Y) + Flux',
        'Gov': 'Warning: Gov=X out of range; should be: Pop(=Y) + Flux',
        'TL': 'Warning: TL=X out of range; should be: mods(=Y) + 1D',
        'Atm': 'Warning: Atm=X out of range; should be: 0...F',
        'Hyd': 'Warning: Hyd=X out of range; should be: 0...A',
        'Worlds': 'Warning: Worlds: W=X out of range',
        'ResLo': 'Warning: (Ex) Resources=X out of range; should be: 2D if TL(=Y)<8',
        'ResHi': 'Warning: (Ex) Resources=X out of range; should be: 2D + GG(=Y) + Belts(=Z) if TL(=A)=8+',
    }

    for sector_file in sectors_list:
        foo = SectorCanonicaliser(sector_file, args.outputdir)
        filename = os.path.basename(sector_file)
        logger.critical('Processing ' + sector_file)
        foo.process()
        foo.write()
        result, lines = foo.lintsec(args.lintsec)
        if not result and 0 < len(lines):
            logger.critical('Warnings found for ' + sector_file)
            out_name = os.path.join(args.outputdir, filename) + '-warnings'
            with codecs.open(out_name, 'w', 'utf-8') as handle:
                for item in lines:
                    bitz = item.split(',')[0]
                    if 'Nonstandard sophont' not in bitz:
                        if 'out of range' in bitz:
                            if 'Law=' in bitz:
                                bitz = out_of_range_warnings['Law']
                            elif 'Gov=' in bitz:
                                bitz = out_of_range_warnings['Gov']
                            elif 'TL=' in bitz:
                                bitz = out_of_range_warnings['TL']
                            elif 'Atm=' in bitz:
                                bitz = out_of_range_warnings['Atm']
                            elif 'Worlds: W=' in bitz:
                                bitz = out_of_range_warnings['Worlds']
                            elif 'Resources=' in bitz and '2D if TL(' in bitz:
                                bitz = out_of_range_warnings['ResLo']
                            elif 'Resources=' in bitz and '2D + GG(' in bitz:
                                bitz = out_of_range_warnings['ResHi']
                            elif 'Hyd=' in bitz:
                                bitz = out_of_range_warnings['Hyd']

                        elif 'Unexpected value for Stellar' in bitz:
                            bitz = 'Warning: Unexpected value for Stellar'
                        elif 'Invalid stellar data:' in bitz:
                            bitz = 'Warning: Invalid stellar data'
                        elif 'PBG: Pop Exponent = 0 but Population Multiplier' in bitz:
                            bitz = 'Warning: PBG: Pop Exponent = 0 but Population Multiplier (=X) > 0'
                        elif 'Warning: {Ix} Importance=' in bitz:
                            bitz = 'Warning: {Ix} Importance=X incorrect; should be: Y'

                        if bitz not in warning_summary:
                            warning_summary[bitz] = 0
                        warning_summary[bitz] += 1

                    item = item.strip() + '\n'
                    handle.write(item)
        logger.critical('Processed ' + sector_file)
        counter += 1

    if 0 < len(warning_summary):
        warning_summary = {k: v for k, v in sorted(warning_summary.items(), key=lambda line: line[1], reverse=True)}
        out_name = os.path.join(args.outputdir, 'warnings-summary')
        with codecs.open(out_name, 'w', 'utf-8') as handle:
            for item in warning_summary:
                line = item + ': ' + str(warning_summary[item]) + '\n'
                handle.write(line)

    logger.critical("%s sectors canonicalised" % counter)


if __name__ == '__main__':
    process()
