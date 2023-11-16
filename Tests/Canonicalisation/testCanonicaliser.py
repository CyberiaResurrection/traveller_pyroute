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
        # Copy of New Vision starline in dictionary should not be TL-canonical
        targ = [item for item in foo.dictionary.lines if 'New Vision' in item][0]
        self.assertTrue('E4248DA-7' in targ, "New Vision starline TL-canonicalised ahead of time")

        foo.write()
        # Verify switch
        targ = [item for item in foo.dictionary.lines if 'New Vision' in item][0]
        self.assertTrue('E4248DA-6' in targ, "New Vision starline not TL-canonicalised")


if __name__ == '__main__':
    unittest.main()
