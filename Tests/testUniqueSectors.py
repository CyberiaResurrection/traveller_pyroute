"""
Created on Jan 12, 2025

@author: CyberiaResurrection
"""
import codecs
import os
import tempfile

from Tests.baseTest import baseTest
from pytest_console_scripts import ScriptRunner


class testUniqueSectors(baseTest):

    def testSectorListIsUnique(self) -> None:
        fullpath = self.unpack_filename('../PyRoute/unique_sectors.py')

        cases = [
            ("sectorlist.txt", "../sectorlist.txt"),
            ("sectorimplist.txt", "../sectorimplist.txt")
        ]

        for filename, fullname in cases:
            with self.subTest(filename):
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as outfile:
                    srcfile = self.unpack_filename(fullname)

                    cwd = os.getcwd()
                    runner = ScriptRunner(launch_mode="subprocess", rootdir=cwd, print_result=False)

                    foo = runner.run([fullpath, "--infile", srcfile, "--outfile", outfile.name])
                    self.assertEqual(0, foo.returncode, "unique_sectors did not complete successfully: " + foo.stderr)

                    inlines = []
                    with codecs.open(srcfile, mode="r", encoding="utf-8") as infile:
                        inlines = [line for line in infile]

                    outlines = []
                    with codecs.open(outfile.name, mode="r", encoding="utf-8") as outfile_foo:
                        outlines = [line for line in outfile_foo]

                self.assertEqual(len(outlines), len(inlines), filename + " has at least one duplicate")
                self.assertEqual(outlines, inlines, "Lines in " + filename + " not correctly sorted")
