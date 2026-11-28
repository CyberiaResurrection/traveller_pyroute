"""
Created on Jan 19, 2025

@author: CyberiaResurrection
"""

import os


class UnpackFilename(object):

    @staticmethod
    def unpack_filename(filename: str) -> str:
        prepend: list[str] = ['../', '../../', 'Tests/', '../Tests/', '../../Tests/', 'Tests/Tests/', '../Tests/Tests/',
                              '../../Tests/Tests/']

        # try unpacked filename directly
        sourcefile = os.path.abspath(filename)
        if not os.path.isfile(sourcefile):
            for item in prepend:
                sourcefile = os.path.abspath(item + filename)
                if os.path.isfile(sourcefile):
                    break

        if not os.path.isfile(sourcefile):
            raise BaseException(sourcefile + " not mapped to a file")

        return sourcefile
