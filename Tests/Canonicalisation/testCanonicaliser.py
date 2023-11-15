import tempfile
import unittest

from PyRoute.Canonicalisation.Canonicaliser import Canonicaliser
from Tests.baseTest import baseTest


class testCanonicaliser(baseTest):
    def test_process(self):
        zaru = self.unpack_filename('DeltaFiles/Zarushagar.sec')

        output_dir = tempfile.gettempdir()
        foo = Canonicaliser(zaru, output_dir)
        foo.process()
        foo.write()


if __name__ == '__main__':
    unittest.main()
