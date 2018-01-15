#! /usr/local/env python3

"""quick script to start a project"""

import sys
from HST.timeSeries import initiateProject

iniFile = sys.argv[1]
initiateProject(iniFile)
