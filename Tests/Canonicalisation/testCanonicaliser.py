import tempfile
import unittest

import requests_mock

from PyRoute.Canonicalisation.Canonicaliser import Canonicaliser
from Tests.baseTest import baseTest


class testCanonicaliser(baseTest):
    def test_process_with_warnings(self):
        content = 'Hint: Parsing as: T5 Second Survey - Column Delimited\r\nWarning: Extraneous code: Ba, line 246: 0917 Deyis II             E874000-2 Ba Da Di(Kebkh) Re                    { -3 } (200-5) [0000] -     -  A 000 10 ImDi K4 II\r\n0 errors, 1 warnings.\r\n\r\n'

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

        url = 'https://travellermap.com/api/sec?lint=1&type=SecondSurvey&hide-tl=0&hide-uwp=0&hide-cap=0'
        with requests_mock.Mocker() as m:
            m.post(url, text=content, status_code=400)
            is_clean, result = foo.lintsec(url)
            self.assertFalse(is_clean)
            expected_result = [
                'Warning: Extraneous code: Ba, line 246: 0917 Deyis II             E874000-2 Ba Da Di(Kebkh) Re                    { -3 } (200-5) [0000] -     -  A 000 10 ImDi K4 II'
            ]
            self.assertEqual(expected_result, result)


if __name__ == '__main__':
    unittest.main()
