import ast
import os
import re
import unittest
from PIL import Image

import networkx as nx
import numpy as np
import pytest
from pymupdf import pymupdf

from PyRoute.DeltaDebug.DeltaDictionary import DeltaDictionary, SectorDictionary
from PyRoute.DeltaDebug.DeltaGalaxy import DeltaGalaxy
from PyRoute.Outputs.ClassicModePDFSectorMap import ClassicModePDFSectorMap
from PyRoute.Outputs.SubsectorMap import SubsectorMap
from PyRoute.Outputs.SubsectorMap2 import GraphicSubsectorMap
from PyRoute.Position.Hex import Hex
from PyRoute.SpeculativeTrade import SpeculativeTrade
from Tests.baseTest import baseTest


class testHexMap(baseTest):
    timestamp_regex = rb'(\d{14,})'
    md5_regex = rb'([0-9a-f]{32,})'
    timeline = re.compile(timestamp_regex)
    md5line = re.compile(md5_regex)

    def test_document_object(self):
        sourcefile = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh empty.sec')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.subsectors = True

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        secname = 'Zao Kfeng Ig Grilokh'

        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")

        blurb = [
            ("Live map", True, os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector.pdf')),
        ]

        for msg, is_live, expected_path in blurb:
            with self.subTest(msg):
                document = hexmap.document(galaxy.sectors[secname], is_live=is_live)
                doc_info = document._doc.info
                self.assertEqual('Sector Zao Kfeng Ig Grilokh (-2,4)', doc_info.title)
                self.assertEqual('Trade route map generated by PyRoute for Traveller', doc_info.subject)
                self.assertEqual('ReportLab', doc_info.creator)
                self.assertEqual(expected_path, document._filename)

    def test_document_object_pdf(self):
        sourcefile = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh empty.sec')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.subsectors = True

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        secname = 'Zao Kfeng Ig Grilokh'

        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")

        blurb = [
            ("Live map", True, os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector.pdf')),
        ]

        for msg, is_live, expected_path in blurb:
            with self.subTest(msg):
                document = hexmap.document(galaxy.sectors[secname], is_live=is_live)
                doc_info = document._doc.info
                self.assertEqual('Sector Zao Kfeng Ig Grilokh (-2,4)', doc_info.title)
                self.assertEqual('Trade route map generated by PyRoute for Traveller', doc_info.subject)
                self.assertEqual('ReportLab', doc_info.creator)
                self.assertEqual(expected_path, document._filename)

    def test_verify_empty_sector_write_pdf(self):
        sourcefile = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh empty.sec')
        srcpdf = self.unpack_filename(
            'OutputFiles/verify_empty_sector_write/Zao Kfeng Ig Grilokh empty.pdf')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.subsectors = True

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)

        galaxy.output_path = args.output

        secname = 'Zao Kfeng Ig Grilokh'

        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")

        targpath = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector.pdf')
        result = hexmap.write_sector_map(galaxy.sectors[secname])
        src_img = pymupdf.open(srcpdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap()
        srcfile = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector original.png')
        src.save(srcfile)
        trg_img = pymupdf.open(targpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap()
        trgfile = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector remix.png')
        trg.save(trgfile)

        image1 = Image.open(srcfile)
        image2 = Image.open(trgfile)

        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.2 > mse, "Image difference " + str(mse) + " above threshold for Zao Kfeng Ig Grilokh sector")

    def test_verify_subsector_trade_write_pdf(self):
        sourcefile = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh - subsector P.sec')
        srcpdf = self.unpack_filename('OutputFiles/verify_subsector_trade_write/Zao Kfeng Ig Grilokh - subsector P - trade.pdf')

        starsfile = self.unpack_filename('OutputFiles/verify_subsector_trade_write/trade stars.txt')
        rangesfile = self.unpack_filename('OutputFiles/verify_subsector_trade_write/trade ranges.txt')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.subsectors = True
        args.routes = 'trade'

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()

        with open(starsfile, 'rb') as file:
            galaxy.stars = nx.read_edgelist(file, nodetype=int)
        self.assertEqual(26, len(galaxy.stars.nodes()), "Unexpected number of stars nodes")
        self.assertEqual(53, len(galaxy.stars.edges), "Unexpected number of stars edges")
        for item in galaxy.stars.edges(data=True):
            self.assertIn('trade', item[2], 'Trade value not set during edgelist read')

        self._load_ranges(galaxy, rangesfile)
        self.assertEqual(27, len(galaxy.ranges.nodes()), "Unexpected number of ranges nodes")
        self.assertEqual(44, len(galaxy.ranges.edges), "Unexpected number of ranges edges")

        secname = 'Zao Kfeng Ig Grilokh'

        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")

        targpath = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector.pdf')
        result = hexmap.write_sector_map(galaxy.sectors[secname])
        src_img = pymupdf.open(srcpdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap()
        srcfile = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector original.png')
        src.save(srcfile)
        trg_img = pymupdf.open(targpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap()
        trgfile = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector remix.png')
        trg.save(trgfile)

        image1 = Image.open(srcfile)
        image2 = Image.open(trgfile)

        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.38 > mse, "Image difference " + str(mse) + " above threshold for Zao Kfeng Ig Grilokh sector")

        graphMap = SubsectorMap(galaxy, args.routes, galaxy.output_path)
        graphMap.write_maps()

    def test_verify_subsector_comm_write_pdf(self):
        sourcefile = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh - subsector P.sec')
        srcpdf = self.unpack_filename(
            'OutputFiles/verify_subsector_comm_write/Zao Kfeng Ig Grilokh - subsector P - comm.pdf')

        starsfile = self.unpack_filename('OutputFiles/verify_subsector_comm_write/comm stars.txt')
        rangesfile = self.unpack_filename('OutputFiles/verify_subsector_comm_write/comm ranges.txt')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'comm'
        args.subsectors = True

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()

        with open(starsfile, 'rb') as file:
            galaxy.stars = nx.read_edgelist(file, nodetype=int)
        self.assertEqual(5, len(galaxy.stars.nodes()), "Unexpected number of stars nodes")
        self.assertEqual(4, len(galaxy.stars.edges), "Unexpected number of stars edges")
        for item in galaxy.stars.edges(data=True):
            self.assertIn('trade', item[2], 'Trade value not set during edgelist read')

        self._load_ranges(galaxy, rangesfile)
        self.assertEqual(27, len(galaxy.ranges.nodes()), "Unexpected number of ranges nodes")
        self.assertEqual(36, len(galaxy.ranges.edges), "Unexpected number of ranges edges")

        secname = 'Zao Kfeng Ig Grilokh'

        hexmap = ClassicModePDFSectorMap(galaxy, 'comm', args.output, "dense")

        targpath = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector.pdf')
        result = hexmap.write_sector_map(galaxy.sectors[secname])
        src_img = pymupdf.open(srcpdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap(dpi=144)
        srcfile = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector original.png')
        src.save(srcfile)
        trg_img = pymupdf.open(targpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap(dpi=144)
        trgfile = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector remix.png')
        trg.save(trgfile)

        image1 = Image.open(srcfile)
        image2 = Image.open(trgfile)

        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.2 > mse, "Image difference " + str(mse) + " above threshold for Zao Kfeng Ig Grilokh sector")

    def test_verify_coreward_rimward_sector(self):
        source1file = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh empty.sec')
        source2file = self.unpack_filename('DeltaFiles/no_subsectors_named/Ngathksirz empty.sec')

        source1pdf = self.unpack_filename('OutputFiles/verify_coreward_rimward/Zao Kfeng Ig Grilokh Sector.pdf')
        source2pdf = self.unpack_filename('OutputFiles/verify_coreward_rimward/Ngathksirz Sector.pdf')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'trade'
        args.subsectors = False

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(source1file)
        delta[sector.name] = sector
        sector = SectorDictionary.load_traveller_map_file(source2file)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()

        zaokpath = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector.pdf')
        ngatpath = os.path.abspath(args.output + '/Ngathksirz Sector.pdf')

        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")

        secname = 'Zao Kfeng Ig Grilokh'
        hexmap.write_sector_map(galaxy.sectors[secname])
        secname = 'Ngathksirz'
        hexmap.write_sector_map(galaxy.sectors[secname])

        srczaok = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector original.png')
        srcngat = os.path.abspath(args.output + '/Ngathksirz Sector original.png')
        trgzaok = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector remix.png')
        trgngat = os.path.abspath(args.output + '/Ngathksirz Sector remix.png')

        src_img = pymupdf.open(source1pdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap(dpi=144)
        src.save(srczaok)

        src_img = pymupdf.open(source2pdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap(dpi=144)
        src.save(srcngat)

        trg_img = pymupdf.open(zaokpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap(dpi=144)
        trg.save(trgzaok)

        trg_img = pymupdf.open(ngatpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap(dpi=144)
        trg.save(trgngat)

        image1 = Image.open(srczaok)
        image2 = Image.open(trgzaok)

        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.2 > mse, "Image difference " + str(mse) + " above threshold for Zao Kfeng Ig Grilokh sector")

        image1 = Image.open(srcngat)
        image2 = Image.open(trgngat)
        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.2 > mse, "Image difference " + str(mse) + " above threshold for Ngathksirz sector")

    def test_verify_spinward_trailing_sector(self):
        source1file = self.unpack_filename('DeltaFiles/no_subsectors_named/Zao Kfeng Ig Grilokh empty.sec')
        source2file = self.unpack_filename('DeltaFiles/no_subsectors_named/Knaeleng empty.sec')

        source1pdf = self.unpack_filename('OutputFiles/verify_spinward_trailing/Zao Kfeng Ig Grilokh Sector.pdf')
        source2pdf = self.unpack_filename('OutputFiles/verify_spinward_trailing/Knaeleng Sector.pdf')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'trade'
        args.subsectors = False

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(source1file)
        delta[sector.name] = sector
        sector = SectorDictionary.load_traveller_map_file(source2file)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()

        zaokpath = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector.pdf')
        ngatpath = os.path.abspath(args.output + '/Knaeleng Sector.pdf')

        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")

        secname = 'Zao Kfeng Ig Grilokh'
        hexmap.write_sector_map(galaxy.sectors[secname])
        secname = 'Knaeleng'
        hexmap.write_sector_map(galaxy.sectors[secname])

        srczaok = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector original.png')
        srcngat = os.path.abspath(args.output + '/Knaeleng Sector original.png')
        trgzaok = os.path.abspath(args.output + '/Zao Kfeng Ig Grilokh Sector remix.png')
        trgngat = os.path.abspath(args.output + '/Knaeleng Sector remix.png')

        src_img = pymupdf.open(source1pdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap(dpi=144)
        src.save(srczaok)

        src_img = pymupdf.open(source2pdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap(dpi=144)
        src.save(srcngat)

        trg_img = pymupdf.open(zaokpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap(dpi=144)
        trg.save(trgzaok)

        trg_img = pymupdf.open(ngatpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap(dpi=144)
        trg.save(trgngat)

        image1 = Image.open(srczaok)
        image2 = Image.open(trgzaok)

        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.2 > mse, "Image difference " + str(mse) + " above threshold for Zao Kfeng Ig Grilokh sector")

        image1 = Image.open(srcngat)
        image2 = Image.open(trgngat)
        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.2 > mse, "Image difference " + str(mse) + " above threshold for Knaeleng sector")

    def test_verify_xboat_write_pdf(self):
        sourcefile = self.unpack_filename('DeltaFiles/Zarushagar-Ibara.sec')
        srcpdf = self.unpack_filename(
            'OutputFiles/verify_subsector_xroute_write/Zarushagar Sector.pdf')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'xroute'
        args.subsectors = False

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()

        secname = 'Zarushagar'

        hexmap = ClassicModePDFSectorMap(galaxy, 'xroute', args.output, "dense")

        targpath = os.path.abspath(args.output + '/Zarushagar Sector.pdf')
        result = hexmap.write_sector_map(galaxy.sectors[secname])
        src_img = pymupdf.open(srcpdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap(dpi=144)
        srcfile = os.path.abspath(args.output + '/Zarushagar Sector original.png')
        src.save(srcfile)
        trg_img = pymupdf.open(targpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap(dpi=144)
        trgfile = os.path.abspath(args.output + '/Zarushagar Sector remix.png')
        trg.save(trgfile)

        image1 = Image.open(srcfile)
        image2 = Image.open(trgfile)

        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.1 > mse, "Image difference " + str(mse) + " above threshold for Zarushagar sector")

    def test_verify_quadripoint_trade_write(self):
        source1file = self.unpack_filename('DeltaFiles/quadripoint_trade_write/Tuglikki.sec')
        source2file = self.unpack_filename('DeltaFiles/quadripoint_trade_write/Provence.sec')
        source3file = self.unpack_filename('DeltaFiles/quadripoint_trade_write/Deneb.sec')
        source4file = self.unpack_filename('DeltaFiles/quadripoint_trade_write/Corridor.sec')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'trade'
        args.subsectors = False
        args.btn = 7

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(source1file)
        delta[sector.name] = sector
        sector = SectorDictionary.load_traveller_map_file(source2file)
        delta[sector.name] = sector
        sector = SectorDictionary.load_traveller_map_file(source3file)
        delta[sector.name] = sector
        sector = SectorDictionary.load_traveller_map_file(source4file)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_routes()

        secname = ['Tuglikki', 'Provence', 'Deneb', 'Corridor']

        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")
        for sector_name in secname:
            hexmap.write_sector_map(galaxy.sectors[sector_name])

        fullname = ['Tuglikki Sector', 'Provence Sector', 'Deneb Sector', 'Corridor Sector']
        srcstem = self.unpack_filename('OutputFiles/verify_quadripoint_trade_write/Corridor Sector.pdf')
        srcstem = srcstem[:-20]
        trgstem = os.path.abspath(args.output + '/')

        for full in fullname:
            srcpdf = srcstem + '/' + full + '.pdf'
            trgpdf = trgstem + '/' + full + '.pdf'

            src_img = pymupdf.open(srcpdf)
            src_iter = src_img.pages(0)
            for page in src_iter:
                src = page.get_pixmap(dpi=144)

            srcfile = os.path.abspath(args.output + '/' + full + ' original.png')
            src.save(srcfile)

            trg_img = pymupdf.open(trgpdf)
            trg_iter = trg_img.pages(0)
            for page in trg_iter:
                trg = page.get_pixmap(dpi=144)
            trgfile = os.path.abspath(args.output + '/' + full + ' remix.png')
            trg.save(trgfile)

            image1 = Image.open(srcfile)
            image2 = Image.open(trgfile)

            array1 = np.array(image1)
            array2 = np.array(image2)

            mse = np.mean((array1 - array2) ** 2)
            self.assertTrue(0.341 > mse, "Image difference " + str(mse) + " above threshold for " + full)

            graphMap = SubsectorMap(galaxy, args.routes, galaxy.output_path)
            graphMap.write_maps()

    def test_verify_single_system_border_write(self):
        sourcefile = self.unpack_filename('DeltaFiles/single_system_border/Deneb.sec')
        srcpdf = self.unpack_filename(
            'OutputFiles/verify_single_system_border_write/Deneb Sector.pdf')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'trade'
        args.subsectors = False
        args.btn = 7

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output
        galaxy.debug_flag = True

        hexbase = galaxy.sectors["Deneb"].worlds[0].hex
        # Hexen at 5 o'clock and 7 o'clock positions on the 2-hex ring
        hex1 = Hex(galaxy.sectors["Deneb"], "0920")
        hex2 = Hex(galaxy.sectors["Deneb"], "1120")
        self.assertEqual(2, hexbase.hex_distance(hex1), "7 o'clock hex should be 2 pc away from base hex")
        self.assertEqual(2, hexbase.hex_distance(hex2), "5 o'clock hex should be 2 pc away from base hex")
        pos1 = (hex1.q, hex1.r)
        pos2 = (hex2.q, hex2.r)
        # Hexen at lower left of 7 o'clock and lower right of 5 o'clock positions, on the 3-hex ring
        hex1o = Hex(galaxy.sectors["Deneb"], "0820")
        hex2o = Hex(galaxy.sectors["Deneb"], "1220")
        self.assertEqual(3, hexbase.hex_distance(hex1o), "7 o'clock outboard hex should be 3 pc away from base hex")
        self.assertEqual(3, hexbase.hex_distance(hex2o), "5 o'clock outboard hex should be 3 pc away from base hex")
        pos1o = (hex1o.q, hex1o.r)
        pos2o = (hex2o.q, hex2o.r)

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_routes()

        expected_ally_map = {
            (-89, 105): 'Im', (-89, 106): 'Im', (-89, 107): 'Im', (-88, 104): 'Im', (-88, 105): 'Im', (-88, 106): 'Im',
            (-88, 107): 'Im', (-87, 103): 'Im', (-87, 104): 'Im', (-87, 105): 'Im', (-87, 106): 'Im', (-87, 107): 'Im',
            (-86, 103): 'Im', (-86, 104): 'Im', (-86, 105): 'Im', (-86, 106): 'Im', (-85, 103): 'Im', (-85, 104): 'Im',
            (-85, 105): 'Im'
        }
        self.assertEqual(expected_ally_map, galaxy.borders.allyMap, "Unexpected allyMap value")

        self.assertEqual('Im', galaxy.borders.allyMap[pos1])
        self.assertEqual('Im', galaxy.borders.allyMap[pos2])
        self.assertNotIn(pos1o, expected_ally_map)
        self.assertNotIn(pos2o, expected_ally_map)

        expected_borders = {
            (-89, 104): ['green', 'orange', 'black'], (-89, 105): [None, 'orange', 'black'], (-89, 106): [None, 'orange', 'black'],
            (-89, 107): ['blue', None, None], (-88, 103): ['green', 'yellow', None], (-88, 107): ['blue', None, 'purple'],
            (-87, 102): ['green', None, 'black'], (-87, 106): [None, 'orange', None], (-87, 107): ['blue', None, None],
            (-86, 102): ['green', None, 'pink'], (-86, 106): ['blue', 'maroon', None], (-85, 101): [None, 'red', None],
            (-85, 102): ['green', None, None], (-85, 105): ['blue', None, 'olive'], (-84, 102): [None, 'maroon', 'pink'],
            (-84, 103): [None, 'maroon', 'pink'], (-84, 104): [None, 'maroon', 'pink']
        }
        self.assertEqual(expected_borders, galaxy.borders.borders, "Unexpected borders value")

        secname = ["Deneb"]

        targpath = os.path.abspath(args.output + '/Deneb Sector.pdf')
        hexmap = ClassicModePDFSectorMap(galaxy, 'trade', args.output, "dense")
        for sector_name in secname:
            hexmap.write_sector_map(galaxy.sectors[sector_name])

        src_img = pymupdf.open(srcpdf)
        src_iter = src_img.pages(0)
        for page in src_iter:
            src = page.get_pixmap(dpi=144)
        srcfile = os.path.abspath(args.output + '/Deneb Sector original.png')
        src.save(srcfile)
        trg_img = pymupdf.open(targpath)
        trg_iter = trg_img.pages(0)
        for page in trg_iter:
            trg = page.get_pixmap(dpi=144)
        trgfile = os.path.abspath(args.output + '/Deneb Sector remix.png')
        trg.save(trgfile)

        image1 = Image.open(srcfile)
        image2 = Image.open(trgfile)

        array1 = np.array(image1)
        array2 = np.array(image2)

        mse = np.mean((array1 - array2) ** 2)
        self.assertTrue(0.11 > mse, "Image difference " + str(mse) + " above threshold for Deneb sector")

    def test_verify_subsector_trade_lines_on_map(self):
        sourcefile = self.unpack_filename('DeltaFiles/verify_subsector_trade_lines_on_map/Trojan Reach.sec')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'trade'
        args.subsectors = False

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output
        galaxy.debug_flag = True

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_routes()

        spectrade = SpeculativeTrade(args.speculative_version, galaxy.stars)
        spectrade.process_tradegoods()

        graphMap = SubsectorMap(galaxy, args.routes, galaxy.output_path)
        graphMap.write_maps()

    def test_verify_subsector_xroute_lines_on_map(self):
        sourcefile = self.unpack_filename('DeltaFiles/verify_subsector_trade_lines_on_map/Trojan Reach.sec')

        args = self._make_args()
        args.interestingline = None
        args.interestingtype = None
        args.maps = True
        args.routes = 'xroute'
        args.subsectors = False

        delta = DeltaDictionary()
        sector = SectorDictionary.load_traveller_map_file(sourcefile)
        delta[sector.name] = sector

        galaxy = DeltaGalaxy(args.btn, args.max_jump)
        galaxy.read_sectors(delta, args.pop_code, args.ru_calc,
                            args.route_reuse, args.routes, args.route_btn, args.mp_threads, args.debug_flag)
        galaxy.output_path = args.output
        galaxy.debug_flag = True

        galaxy.generate_routes()
        galaxy.set_borders(args.borders, args.ally_match)
        galaxy.trade.calculate_routes()

        spectrade = SpeculativeTrade(args.speculative_version, galaxy.stars)
        spectrade.process_tradegoods()

        graphMap = SubsectorMap(galaxy, args.routes, galaxy.output_path)
        graphMap.write_maps()

    def _load_ranges(self, galaxy, sourcefile):
        with open(sourcefile, "rb") as f:
            lines = f.readlines()
            for rawline in lines:
                line = rawline.strip()
                bitz = line.split(b') ')
                source = str(bitz[0]).replace('\'', '').lstrip('b')
                target = str(bitz[1]).replace('\'', '').lstrip('b')
                srcbitz = source.split('(')
                targbitz = target.split('(')
                hex1 = srcbitz[1][-4:]
                sec1 = srcbitz[1][0:-5]
                hex2 = targbitz[1][-4:]
                sec2 = targbitz[1][0:-5]

                world1 = galaxy.sectors[sec1].find_world_by_pos(hex1)
                world2 = galaxy.sectors[sec2].find_world_by_pos(hex2)
                rawdata = str(bitz[2]).lstrip('b')
                data = ast.literal_eval(ast.literal_eval(rawdata))

                galaxy.ranges.add_edge(world1, world2)
                for item in data:
                    galaxy.ranges[world1][world2][item] = data[item]


if __name__ == '__main__':
    unittest.main()
